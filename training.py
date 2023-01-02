import copy
import datetime
import itertools
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def apply_spectral_normalization(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, (nn.modules.conv._ConvNd, nn.Linear)):
            setattr(module, name, torch.nn.utils.parametrizations.spectral_norm(child))
        else:
            apply_spectral_normalization(child)


def disable_bn_running_stats(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.track_running_stats = False


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


@dataclass
class GANTrainerConfig:
    conditional: bool = False
    z_dim: int = 128
    method: Literal["gan", "wgan", "wgan-gp", "hinge"] = "gan"
    spectral_norm_d: bool = False
    spectral_norm_g: bool = False
    label_smoothing: float = 0.0
    clip: float = 0.01
    lamb: float = 10.0
    train_g_interval: int = 1
    optimizer: Literal["SGD", "Adam", "AdamW", "RMSprop"] = "Adam"
    lr: float = 2e-4
    lr_d: Optional[float] = None
    lr_g: Optional[float] = None
    weight_decay: float = 0
    beta1: float = 0.5
    beta2: float = 0.999
    n_steps: int = 10000
    ema: bool = False
    ema_decay: float = 0.999
    log_name: str = "gan"
    log_img_interval: int = 1000


class GANTrainer:
    def __init__(self, config: GANTrainerConfig, D: nn.Module, G: nn.Module, fixed_z: Tensor, fixed_y: Tensor):
        assert config.clip > 0
        config = copy.deepcopy(config)
        config.lr_d = config.lr_d or config.lr
        config.lr_g = config.lr_g or config.lr
        config.log_name += "/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        accelerator = Accelerator(log_with="tensorboard", logging_dir="logs")
        rank_zero = accelerator.is_main_process

        if config.spectral_norm_d:
            apply_spectral_normalization(D)

        # spectral norm and Accelerate's fp32 wrapper will throw error to copy.deepcopy
        G_ema = EMA(G, config.ema_decay, device=accelerator.device) if config.ema and rank_zero else None
        if config.spectral_norm_g:
            apply_spectral_normalization(G)
            if G_ema is not None:
                apply_spectral_normalization(G_ema.ema_model)

        if accelerator.distributed_type == "MULTI_GPU":
            disable_bn_running_stats(D)
            disable_bn_running_stats(G)

        optim_d, optim_g = self.build_optimizers(config, D, G)
        D, G, optim_d, optim_g = accelerator.prepare(D, G, optim_d, optim_g)

        accelerator.init_trackers(config.log_name, vars(config))
        accelerator.print(config, AcceleratorState(), D, G, sep="\n")
        accelerator.print(f"D: {count_params(D)/1e6:.2f}M params")
        accelerator.print(f"G: {count_params(G)/1e6:.2f}M params")

        self.config = config
        self.accelerator = accelerator
        self.D = D
        self.G = G
        self.G_ema = G_ema
        self.optim_d = optim_d
        self.optim_g = optim_g
        self.fixed_z = fixed_z.to(accelerator.device) if rank_zero else None
        self.fixed_y = fixed_y.to(accelerator.device) if rank_zero else None

    @staticmethod
    def build_optimizers(config: GANTrainerConfig, D: nn.Module, G: nn.Module):
        optim_cls = getattr(torch.optim, config.optimizer)
        kwargs = dict(weight_decay=config.weight_decay)
        if config.optimizer in ("Adam", "AdamW"):
            kwargs.update(betas=(config.beta1, config.beta2))

        optim_d = optim_cls(D.parameters(), lr=config.lr_d, **kwargs)

        if hasattr(G, "mapping_network"):  # stylegan
            group1 = [p for name, p in G.named_parameters() if not name.startswith("mapping_network.")]
            group2 = list(G.mapping_network.parameters())
            assert len(group1) + len(group2) == len(list(G.parameters()))
            param_groups = [
                dict(params=group1),
                dict(params=group2, lr=config.lr_g / 100),
            ]
            optim_g = optim_cls(param_groups, **kwargs)
        else:
            param_groups = [dict(params=list(G.parameters()))]
        optim_g = optim_cls(param_groups, lr=config.lr_g, **kwargs)

        return optim_d, optim_g

    def train(self, dloader: DataLoader):
        dloader = self.accelerator.prepare(dloader)
        step, finished = 0, False
        self.log_images(step)

        for epoch in itertools.count():
            pbar = tqdm(dloader, desc=f"Epoch {epoch}", leave=False) if self.accelerator.is_main_process else dloader
            for x_reals, ys in pbar:
                step += 1
                log_dict = dict(epoch=epoch)

                loss_d = self.train_D_step(x_reals, ys)
                log_dict["loss/d"] = loss_d.item()

                if step % self.config.train_g_interval == 0:
                    loss_g = self.train_G_step(x_reals, ys)
                    log_dict["loss/g"] = loss_g.item()

                    if self.G_ema is not None:
                        self.G_ema.update()

                self.accelerator.log(log_dict, step=step)
                if step % self.config.log_img_interval == 0:
                    self.log_images(step)

                if step >= self.config.n_steps:
                    finished = True
                    break

            if finished:
                break

        self.accelerator.end_training()

    def _forward(self, m: nn.Module, xs: Tensor, ys: Tensor) -> Tensor:
        return m(xs, ys) if self.config.conditional else m(xs)

    def train_D_step(self, x_reals: Tensor, ys: Tensor):
        bsize = x_reals.shape[0]
        method = self.config.method

        with torch.no_grad():
            z_noise = torch.randn(bsize, self.config.z_dim, device=self.accelerator.device)
            x_fakes = self._forward(self.G, z_noise, ys)
        d_reals = self._forward(self.D, x_reals, ys)
        d_fakes = self._forward(self.D, x_fakes, ys)

        if method == "gan":
            loss_d_real = -F.logsigmoid(d_reals).mean() * (1.0 - self.config.label_smoothing)
            loss_d_fake = -F.logsigmoid(-d_fakes).mean()
            loss_d = loss_d_real + loss_d_fake

        elif method in ("wgan", "wgan-gp"):
            loss_d = d_fakes.mean() - d_reals.mean()

            if method == "wgan-gp":
                alpha = torch.rand(bsize, 1, 1, 1, device=x_reals.device)
                x_inters = (x_reals * alpha + x_fakes * (1 - alpha)).requires_grad_()
                d_inters = self._forward(self.D, x_inters, ys)

                (d_grad,) = torch.autograd.grad(d_inters, x_inters, torch.ones_like(d_inters), create_graph=True)
                d_grad_norm = torch.linalg.vector_norm(d_grad, dim=(1, 2, 3))
                loss_d = loss_d + self.config.lamb * ((d_grad_norm - 1) ** 2).mean()

        elif method == "hinge":
            loss_d = F.relu(1 - d_reals).mean() + F.relu(1 + d_fakes).mean()

        else:
            raise ValueError(f"Unsupported method {method}")

        self.optim_d.zero_grad(set_to_none=True)
        self.accelerator.backward(loss_d)
        self.optim_d.step()

        # Algorithm 1 in paper clip weights after optimizer step, but GitHub code clip before optimizer step
        # it doesn't seem to matter much in practice
        if method == "wgan":
            clip = self.config.clip
            with torch.no_grad():
                for param in self.D.parameters():
                    param.clip_(-clip, clip)

        return loss_d

    def train_G_step(self, x_reals: Tensor, ys: Tensor):
        bsize = x_reals.shape[0]
        method = self.config.method
        self.D.requires_grad_(False)

        z_noise = torch.randn(bsize, self.config.z_dim, device=self.accelerator.device)
        x_fakes = self._forward(self.G, z_noise, ys)
        d_fakes = self._forward(self.D, x_fakes, ys)

        if method == "gan":
            loss_g = -F.logsigmoid(d_fakes).mean()

        elif method in ("wgan", "wgan-gp", "hinge"):
            loss_g = -d_fakes.mean()

        else:
            raise ValueError(f"Unsupported method {method}")

        self.optim_g.zero_grad(set_to_none=True)
        self.accelerator.backward(loss_g)
        self.optim_g.step()

        self.D.requires_grad_(True)
        return loss_g

    @torch.inference_mode()
    def log_images(self, step: int):
        if not self.accelerator.is_main_process:
            return
        logger = self.accelerator.get_tracker("tensorboard")

        x_fakes = self._forward(self.G, self.fixed_z, self.fixed_y).mul_(0.5).add_(0.5)
        logger.add_images("generated", x_fakes, step)

        if self.G_ema is not None:
            x_fakes = self._forward(self.G_ema, self.fixed_z, self.fixed_y).mul_(0.5).add_(0.5)
            logger.add_images("generated/ema", x_fakes, step)


# reference: https://github.com/lucidrains/ema-pytorch
class EMA(nn.Module):
    def __init__(self, model: nn.Module, ema_decay: float, warmup: int = 100, device=None):
        super().__init__()
        self.model = model
        ema_model = copy.deepcopy(model)
        if device is not None:
            ema_model = ema_model.to(device)
        self.ema_model = ema_model
        self.ema_decay = ema_decay
        self.warmup = warmup
        self.counter = 0

    @torch.no_grad()
    def update(self):
        self.counter += 1
        if self.counter <= self.warmup:
            return

        for name, param in self.model.named_parameters():
            ema_param = self.ema_model.get_parameter(name)
            diff = param - ema_param
            ema_param.add_(diff.mul_(1 - self.ema_decay))

    def forward(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)
