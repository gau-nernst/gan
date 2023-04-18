import contextlib
import copy
import datetime
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def cuda_speed_up():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def apply_spectral_normalization(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, (nn.modules.conv._ConvNd, nn.Linear)):
            setattr(module, name, torch.nn.utils.parametrizations.spectral_norm(child))
        else:
            apply_spectral_normalization(child)


def disable_bn_running_stats(module: nn.Module):
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.track_running_stats = False


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def cycle(iterable):
    while True:
        for item in iterable:
            yield item


@dataclass
class GANTrainerConfig:
    conditional: bool = False
    z_dim: int = 128
    method: Literal["gan", "wgan", "wgan-gp", "hinge"] = "gan"
    spectral_norm_d: bool = False
    spectral_norm_g: bool = False
    label_smoothing: float = 0.0
    wgan_clip: float = 0.01
    wgan_gp_lamb: float = 10.0
    drift_penalty: float = 0.0
    r1_penalty: float = 0.0
    r1_penalty_interval: int = 1
    train_g_interval: int = 1
    optimizer: Literal["SGD", "Adam", "AdamW", "RMSprop"] = "Adam"
    lr: float = 2e-4
    lr_d: Optional[float] = None
    lr_g: Optional[float] = None
    weight_decay: float = 0
    beta1: float = 0.5
    beta2: float = 0.999
    n_steps: int = 10_000
    ema: bool = False
    ema_decay: float = 0.999
    ema_device: Optional[str] = None
    checkpoint_interval: int = 10_000
    checkpoint_path: str = "checkpoints"
    resume_from_checkpoint: Optional[str] = None
    loggers: List[str] = field(default_factory=list)
    log_name: str = "gan"
    log_interval: int = 50
    log_img_interval: int = 1_000
    find_unused_parameters: bool = False


class GANTrainer:
    def __init__(self, config: GANTrainerConfig, D: nn.Module, G: nn.Module, fixed_z: Tensor, fixed_y: Tensor):
        assert config.wgan_clip > 0
        config = copy.deepcopy(config)
        config.lr_d = config.lr_d or config.lr
        config.lr_g = config.lr_g or config.lr

        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config.checkpoint_path += f"/{config.log_name}/{now}"

        cuda_speed_up()
        accelerator = Accelerator(
            log_with=config.loggers,
            project_dir="logs",
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=config.find_unused_parameters)],
        )
        rank_zero = accelerator.is_main_process
        config.ema_device = config.ema_device or str(accelerator.device)

        if config.spectral_norm_d:
            apply_spectral_normalization(D)

        if config.spectral_norm_g:
            apply_spectral_normalization(G)

        # spectral norm and Accelerate's fp32 wrapper will throw error to copy.deepcopy
        G_ema = EMA(G, config.ema_decay, device=accelerator.device, enable=config.ema and rank_zero)
        if config.ema and rank_zero:
            accelerator.register_for_checkpointing(G_ema)

        if accelerator.distributed_type == "MULTI_GPU":
            D.apply(disable_bn_running_stats)
            G.apply(disable_bn_running_stats)

        optim_d, optim_g = self.build_optimizers(config, D, G)
        D, G, optim_d, optim_g = accelerator.prepare(D, G, optim_d, optim_g)

        if config.resume_from_checkpoint is not None:
            accelerator.load_state(config.resume_from_checkpoint)

        _config = {k: str(v) if not isinstance(v, (int, float, bool, str)) else v for k, v in vars(config).items()}
        accelerator.init_trackers(config.log_name, config=_config)
        accelerator.print(config, AcceleratorState(), D, G, optim_d, optim_g, sep="\n")
        accelerator.print(f"D: {count_params(D)/1e6:.2f}M params")
        accelerator.print(f"G: {count_params(G)/1e6:.2f}M params")

        self.config = config
        self.accelerator = accelerator
        self.D = D
        self.G = G
        self.G_ema = G_ema
        self.optim_d = optim_d
        self.optim_g = optim_g
        self.fixed_z = fixed_z.to(accelerator.device)
        self.fixed_y = fixed_y.to(accelerator.device)
        self.counter = 0

    @staticmethod
    def build_optimizers(config: GANTrainerConfig, D: nn.Module, G: nn.Module):
        optim_cls = getattr(torch.optim, config.optimizer)
        kwargs: dict[str, Any] = dict(weight_decay=config.weight_decay)
        if config.optimizer in ("Adam", "AdamW"):
            kwargs.update(betas=(config.beta1, config.beta2))

        optim_d = optim_cls(D.parameters(), lr=config.lr_d, **kwargs)

        if hasattr(G, "mapping_network") and isinstance(G.mapping_network, nn.Module):  # stylegan
            group1 = [p for name, p in G.named_parameters() if not name.startswith("mapping_network.")]
            group2 = list(G.mapping_network.parameters())
            assert len(group1) + len(group2) == len(list(G.parameters()))
            param_groups = [
                dict(params=group1, lr=config.lr_g),
                dict(params=group2, lr=config.lr_g / 100),
            ]
            optim_g = optim_cls(param_groups, **kwargs)
        else:
            param_groups = [dict(params=list(G.parameters()))]
        optim_g = optim_cls(param_groups, lr=config.lr_g, **kwargs)

        return optim_d, optim_g

    def _forward(self, m: nn.Module, xs: Tensor, ys: Optional[Tensor]) -> Tensor:
        return m(xs, ys) if self.config.conditional else m(xs)

    def train(self, dloader: DataLoader):
        cfg = self.config
        dloader = self.accelerator.prepare(dloader)

        pbar = tqdm(
            cycle(dloader),
            total=cfg.n_steps,
            leave=False,
            dynamic_ncols=True,
            disable=not self.accelerator.is_main_process,
        )

        for x_reals, ys in pbar:
            if self.counter % cfg.log_img_interval == 0 and self.accelerator.is_main_process:
                with torch.inference_mode(), self.G_ema.swap_state_dict(self.G):
                    x_fakes = self._forward(self.G, self.fixed_z, self.fixed_y)
                self.log_images(x_fakes, "generated", self.counter)

            log_dict: dict[str, Any] = dict()
            log_dict["loss/d"] = self.train_D_step(x_reals, ys, self.counter).item()

            if self.counter % cfg.train_g_interval == 0:
                log_dict["loss/g"] = self.train_G_step(x_reals, ys, self.counter).item()
                self.G_ema.update(self.G)

            # loss values are associated with models before the optimizer step
            # therefore, increment `step` after logging
            if self.counter % cfg.log_interval == 0:
                self.accelerator.log(log_dict, step=self.counter)

            self.counter += 1

            if self.counter % cfg.checkpoint_interval == 0:
                self.accelerator.save_state(f"{cfg.checkpoint_path}/step_{self.counter:07d}")

            for m in (self.D, self.G):
                if hasattr(m, "step"):
                    m.step()

            if self.counter >= self.config.n_steps:
                break

        # self.accelerator.end_training()

    def train_D_step(self, x_reals: Tensor, ys: Optional[Tensor]):
        bsize = x_reals.shape[0]
        cfg = self.config

        with torch.no_grad():
            z_noise = torch.randn(bsize, cfg.z_dim, device=self.accelerator.device)
            x_fakes = self._forward(self.G, z_noise, ys)

        # for progressive growing
        # F.avg_pool2d() is significantly faster than F.interpolate(mode="bilinear", antialias=True)
        # if x_reals.shape[-2:] != x_fakes.shape[-2:]:
        #     x_reals = F.adaptive_avg_pool2d(x_reals, x_fakes.shape[-2:])

        if cfg.r1_penalty > 0 and self.counter % cfg.r1_penalty_interval == 0:
            x_reals.requires_grad_()

        d_reals = self._forward(self.D, x_reals, ys)
        d_fakes = self._forward(self.D, x_fakes, ys)

        if cfg.method == "gan":
            loss_d_real = -F.logsigmoid(d_reals).mean() * (1.0 - cfg.label_smoothing)
            loss_d_fake = -F.logsigmoid(-d_fakes).mean()
            loss_d = loss_d_real + loss_d_fake

        elif cfg.method in ("wgan", "wgan-gp"):
            loss_d = d_fakes.mean() - d_reals.mean()

            if cfg.method == "wgan-gp":
                alpha = torch.rand(bsize, 1, 1, 1, device=x_reals.device)
                x_inters = x_reals.lerp(x_fakes, alpha).requires_grad_()
                d_inters = self._forward(self.D, x_inters, ys)

                (d_grad,) = torch.autograd.grad(d_inters.sum(), x_inters, create_graph=True)
                d_grad_norm = torch.linalg.vector_norm(d_grad, dim=(1, 2, 3))
                loss_d = loss_d + cfg.wgan_gp_lamb * (d_grad_norm - 1).square().mean()

        elif cfg.method == "hinge":
            loss_d = F.relu(1 - d_reals).mean() + F.relu(1 + d_fakes).mean()

        else:
            raise ValueError(f"Unsupported method {cfg.method}")

        # for Progressive GAN only. may remove?
        if cfg.drift_penalty > 0:
            loss_d = loss_d + d_reals.square().mean() * cfg.drift_penalty

        # https://arxiv.org/abs/1801.04406, for StyleGAN and StyleGAN2
        if cfg.r1_penalty > 0 and self.counter % cfg.r1_penalty_interval == 0:
            (d_grad,) = torch.autograd.grad(d_reals.sum(), x_reals, create_graph=True)
            d_grad_norm2 = d_grad.square().sum() / bsize
            loss_d = loss_d + d_grad_norm2 * cfg.r1_penalty / 2

        self.optim_d.zero_grad(set_to_none=True)
        self.accelerator.backward(loss_d)
        self.optim_d.step()

        # Algorithm 1 in paper clip weights after optimizer step, but GitHub code clip before optimizer step
        # it shouldn't matter much in practice
        if cfg.method == "wgan":
            with torch.no_grad():
                for param in self.D.parameters():
                    param.clip_(-cfg.wgan_clip, cfg.wgan_clip)

        return loss_d

    def train_G_step(self, x_reals: Tensor, ys: Optional[Tensor]):
        bsize = x_reals.shape[0]
        cfg = self.config
        self.D.requires_grad_(False)

        z_noise = torch.randn(bsize, cfg.z_dim, device=self.accelerator.device)
        x_fakes = self._forward(self.G, z_noise, ys)
        d_fakes = self._forward(self.D, x_fakes, ys)

        if cfg.method == "gan":
            loss_g = -F.logsigmoid(d_fakes).mean()

        elif cfg.method in ("wgan", "wgan-gp", "hinge"):
            loss_g = -d_fakes.mean()

        else:
            raise ValueError(f"Unsupported method {cfg.method}")

        self.optim_g.zero_grad(set_to_none=True)
        self.accelerator.backward(loss_g)
        self.optim_g.step()

        self.D.requires_grad_(True)
        return loss_g

    @torch.inference_mode()
    def log_images(self, imgs: Tensor, tag: str, step: int):
        if not self.accelerator.is_main_process or len(self.accelerator.trackers) == 0:
            return

        imgs_np = imgs.cpu().clip_(-1, 1).add_(1).mul_(255 / 2).byte().permute(0, 2, 3, 1).numpy()

        for tracker in self.accelerator.trackers:
            if tracker.name == "tensorboard":
                tracker.tracker.add_images(tag, imgs_np, step, dataformats="NHWC")

            elif tracker.name == "wandb":
                import wandb

                wandb_imgs = [wandb.Image(img) for img in imgs_np]
                tracker.tracker.log({tag: wandb_imgs}, step=step)


# reference: https://github.com/lucidrains/ema-pytorch
class EMA:
    def __init__(
        self,
        model: nn.Module,
        ema_decay: float = 0.999,
        warmup: int = 100,
        device: Optional[torch.device] = None,
        enable: bool = False,
    ):
        super().__init__()
        self.enable = enable
        self.ema_state_dict = {}
        if enable:
            self.ema_decay = ema_decay
            self.warmup = warmup
            self.counter = 0
            for k, v in model.state_dict().items():
                self.ema_state_dict[k] = v.new_empty(v.shape, device=device)
                self.ema_state_dict[k].copy_(v)

    @torch.no_grad()
    def update(self, model: nn.Module):
        if not self.enable:
            return

        self.counter += 1
        if self.counter < self.warmup:
            return

        for k, v in model.state_dict().items():
            ema_v = self.ema_state_dict[k]
            if self.counter == self.warmup or not v.is_floating_point():
                ema_v.copy_(v)
            else:
                ema_v.lerp_(v, 1 - self.ema_decay)

    @staticmethod
    def _swap_state_dict(state_dict_a, state_dict_b):
        for k, v in state_dict_b.items():
            tmp = v.clone()
            v.copy_(state_dict_a[k])
            state_dict_a[k].copy_(tmp)

    @contextlib.contextmanager
    def swap_state_dict(self, model: nn.Module):
        if not self.enable:
            yield
            return

        self._swap_state_dict(self.ema_state_dict, model.state_dict())
        yield
        self._swap_state_dict(self.ema_state_dict, model.state_dict())

    def state_dict(self):
        return {"counter": self.counter, **self.ema_state_dict}

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if k == "counter":
                self.counter = v
            else:
                self.ema_state_dict[k] = v
