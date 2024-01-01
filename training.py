import contextlib
import copy
import datetime
import sys
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def cycle(iterable):
    while True:
        for item in iterable:
            yield item


@dataclass
class BaseTrainerConfig:
    conditional: bool = False
    z_dim: int = 128
    method: Literal["gan", "wgan", "wgan-gp", "hinge", "lsgan"] = "gan"
    spectral_norm_d: bool = False
    spectral_norm_g: bool = False
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
    beta1: float = 0.9
    beta2: float = 0.999
    ema: bool = False
    ema_decay: float = 0.999
    ema_device: Optional[str] = None
    checkpoint_interval: int = 10_000
    checkpoint_path: str = "checkpoints"
    resume_from_checkpoint: Optional[str] = None
    loggers: list[str] = field(default_factory=list)
    log_name: str = "gan"
    log_interval: int = 50
    log_img_interval: int = 1_000
    find_unused_parameters: bool = False
    amp: bool = False
    channels_last: bool = False
    device: Optional[str] = None


class BaseTrainer:
    def __init__(self, config: BaseTrainerConfig, dis: nn.Module, gen: nn.Module):
        assert config.wgan_clip > 0
        config = copy.deepcopy(config)
        config.lr_d = config.lr_d or config.lr
        config.lr_g = config.lr_g or config.lr
        config.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        cuda_speed_up()

        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config.checkpoint_path += f"/{config.log_name}/{now}"

        dis = dis.to(device=config.device)
        gen = gen.to(device=config.device)

        if config.channels_last:
            dis = dis.to(memory_format=torch.channels_last)
            gen = gen.to(memory_format=torch.channels_last)

        if config.spectral_norm_d:
            apply_spectral_normalization(dis)

        if config.spectral_norm_g:
            apply_spectral_normalization(gen)

        self.g_ema = EMA(gen, config.ema_decay, enable=config.ema)
        self.optim_d, self.optim_g = build_optimizers(config, dis, gen)

        print(config, dis, gen, self.optim_d, self.optim_g, sep="\n")
        print(f"D: {count_params(dis)/1e6:.2f}M params")
        print(f"G: {count_params(gen)/1e6:.2f}M params")

        self.config = config
        self.dis = dis
        self.gen = gen
        self.global_step = 0
        self.logger = SummaryWriter(f"logs/{config.log_name}")
        self.scaler = torch.cuda.amp.GradScaler() if config.amp else None

    @contextlib.contextmanager
    def autocast(self, enabled=True):
        if self.config.amp:
            context = torch.autocast(torch.device(self.config.device).type, torch.float16, enabled=enabled)
            context.__enter__()
            yield
            context.__exit__(*sys.exc_info())
        else:
            yield

    def step_optimizer(self, loss: Tensor, optimizer: torch.optim.Optimizer):
        optimizer.zero_grad(set_to_none=True)
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
        else:
            loss.backward()
            optimizer.step()

    def train(self, dloader: DataLoader, n_steps: int):
        cfg = self.config
        total_steps = self.global_step + n_steps
        pbar = tqdm(
            cycle(dloader),
            total=total_steps,
            initial=self.global_step,
            leave=False,
            dynamic_ncols=True,
        )

        for data in pbar:
            if isinstance(data, Tensor):
                data = (data,)
            data = tuple(x.to(cfg.device) for x in data)
            log_dict = self.train_step(*data)

            if self.scaler is not None:
                self.scaler.update()

            # loss values are associated with models before the optimizer step
            # therefore, increment `step` after logging
            if self.global_step % cfg.log_interval == 0:
                for k, v in log_dict.items():
                    self.logger.add_scalar(k, v, self.global_step)

            if "loss/g" in log_dict:
                self.g_ema.update(self.gen)

            self.global_step += 1

            if self.global_step >= total_steps:
                break

    def train_step(self, *args):
        raise NotImplementedError

    @torch.inference_mode()
    def log_images(self, imgs: Tensor, tag: str, step: int):
        imgs = imgs.cpu().float().clip_(-1, 1).add_(1).div_(2)
        imgs_np = imgs.mul_(255).byte().permute(0, 2, 3, 1).numpy()

        self.logger.add_images(tag, imgs_np, step, dataformats="NHWC")
        # wandb_imgs = [wandb.Image(img) for img in imgs_np]
        # wandb.log({tag: wandb_imgs}, step=step)


def compute_g_loss(d_fakes: Tensor, method: str) -> Tensor:
    if method == "gan":
        loss_g = -F.logsigmoid(d_fakes).mean()

    elif method in ("wgan", "wgan-gp", "hinge"):
        loss_g = -d_fakes.mean()

    elif method == "lsgan":
        loss_g = (d_fakes - 1).square().mean()

    else:
        raise ValueError(f"Unsupported method {method}")

    return loss_g


def compute_d_loss(d_reals: Tensor, d_fakes: Tensor, method: str) -> Tensor:
    if method == "gan":
        loss_d = -F.logsigmoid(d_reals).mean() - F.logsigmoid(-d_fakes).mean()

    elif method in ("wgan", "wgan-gp"):
        loss_d = d_fakes.mean() - d_reals.mean()

    elif method == "hinge":
        loss_d = F.relu(1 - d_reals).mean() + F.relu(1 + d_fakes).mean()

    elif method == "lsgan":
        loss_d = (d_reals - 1).square().mean() + d_fakes.square().mean()

    else:
        raise ValueError(f"Unsupported method {method}")

    return loss_d


def build_optimizers(
    config: BaseTrainerConfig, dis: nn.Module, gen: nn.Module
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    kwargs = dict(weight_decay=config.weight_decay)
    if config.optimizer in ("Adam", "AdamW"):
        kwargs.update(betas=(config.beta1, config.beta2))

    optim_cls = getattr(torch.optim, config.optimizer)
    optim_d = optim_cls(dis.parameters(), lr=config.lr_d, **kwargs)

    # stylegan
    if hasattr(gen, "mapping_network") and isinstance(gen.mapping_network, nn.Module):
        group1 = [p for name, p in gen.named_parameters() if not name.startswith("mapping_network.")]
        group2 = list(gen.mapping_network.parameters())
        assert len(group1) + len(group2) == len(list(gen.parameters()))
        param_groups = [
            dict(params=group1, lr=config.lr_g),
            dict(params=group2, lr=config.lr_g / 100),
        ]
        optim_g = optim_cls(param_groups, **kwargs)
    else:
        param_groups = [dict(params=list(gen.parameters()))]

    optim_g = optim_cls(param_groups, lr=config.lr_g, **kwargs)

    return optim_d, optim_g


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
    def update(self, model: nn.Module) -> None:
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
    def _swap_state_dict(state_dict_a, state_dict_b) -> None:
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
