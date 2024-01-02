import argparse
import inspect
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision.io import write_png
from torchvision.transforms import v2
from tqdm import tqdm

from ema import EMA
from modelling.dcgan import DcGanDiscriminator, DcGanGenerator


def unnormalize(x: Tensor) -> Tensor:
    return ((x * 0.5 + 0.5) * 255).round().to(torch.uint8)


def gan_d_loss(disc: nn.Module, reals: Tensor, fakes: Tensor) -> Tensor:
    return -F.logsigmoid(disc(reals)).mean() - F.logsigmoid(-disc(fakes)).mean()


def gan_g_loss(d_fakes: Tensor, disc: nn.Module, reals: Tensor) -> Tensor:
    return -F.logsigmoid(d_fakes).mean()


def wgan_d_loss(disc: nn.Module, reals: Tensor, fakes: Tensor) -> Tensor:
    with torch.no_grad():
        for p in disc.parameters():
            p.clip_(-0.01, 0.01)
    return -disc(reals).mean() + disc(fakes).mean()


def wgan_g_loss(d_fakes: Tensor, disc: nn.Module, reals: Tensor) -> Tensor:
    return -d_fakes.mean()


def rgan_d_loss(disc: nn.Module, reals: Tensor, fakes: Tensor) -> Tensor:
    return -F.logsigmoid(disc(reals) - disc(fakes)).mean()


def rgan_g_loss(d_fakes: Tensor, disc: nn.Module, reals: Tensor) -> Tensor:
    with torch.no_grad():
        d_reals = disc(reals)
    return -F.logsigmoid(d_fakes - d_reals).mean()


@dataclass
class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = False
    n_epochs: int = 10
    lr: float = 2e-4
    batch_size: int = 128
    method: str = "gan"
    log_dir: str = "images_celeba"


def make_parser(fn):
    parser = argparse.ArgumentParser()

    for k, v in inspect.signature(fn).parameters.items():
        if v.annotation in (str, int, float):
            parser.add_argument(f"--{k}", type=v.annotation, default=v.default)
        elif v.annotation is bool:
            parser.add_argument(f"--{k}", action="store_true")
        else:
            raise RuntimeError(f"Unsupported type {v.annotation} for argument {k}")

    return parser


if __name__ == "__main__":
    args = make_parser(TrainConfig).parse_args()
    cfg = TrainConfig(**vars(args))

    print("Config:")
    for k, v in vars(cfg).items():
        print(f"  {k}: {v}")

    disc = DcGanDiscriminator().to(cfg.device)
    gen = DcGanGenerator().to(cfg.device)
    print(disc)
    print(gen)

    gen_ema = EMA(gen)
    d_criterion, g_criterion = {
        "gan": (gan_d_loss, gan_g_loss),
        "wgan": (wgan_d_loss, wgan_g_loss),
        "rgan": (rgan_d_loss, rgan_g_loss),
    }[cfg.method]

    optim_d = torch.optim.AdamW(disc.parameters(), cfg.lr, betas=(0.5, 0.999), weight_decay=0)
    optim_g = torch.optim.AdamW(gen.parameters(), cfg.lr, betas=(0.5, 0.999), weight_decay=0)

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(64, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(64),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    ds = CelebA("data", transform=transform, download=True)
    dloader = DataLoader(ds, cfg.batch_size, shuffle=True, num_workers=4, drop_last=True)

    autocast_ctx = torch.autocast(cfg.device, dtype=torch.bfloat16, enabled=cfg.mixed_precision)
    os.makedirs(cfg.log_dir, exist_ok=True)
    fixed_zs = torch.randn(100, 128, device=cfg.device)
    step = 0

    for epoch_idx in range(cfg.n_epochs):
        for reals, _ in tqdm(dloader):
            reals = reals.to(cfg.device)
            zs = torch.randn(cfg.batch_size, 128, device=cfg.device)
            with autocast_ctx:
                with torch.no_grad():
                    fakes = gen(zs)
                loss_d = d_criterion(disc, reals, fakes)
            loss_d.backward()
            optim_d.step()
            optim_d.zero_grad()

            disc.requires_grad_(False)
            zs = torch.randn(cfg.batch_size, 128, device=cfg.device)
            with autocast_ctx:
                loss_g = g_criterion(disc(gen(zs)), disc, reals)
            loss_g.backward()
            optim_g.step()
            optim_g.zero_grad()
            disc.requires_grad_(True)
            gen_ema.step()

        for suffix, model in [("", gen), ("_ema", gen_ema)]:
            with autocast_ctx, torch.no_grad():
                fakes = model(fixed_zs)
            fakes = fakes.cpu().view(10, 10, 3, 64, 64).permute(2, 0, 3, 1, 4).reshape(3, 640, 640)
            write_png(unnormalize(fakes), f"{cfg.log_dir}/epoch{epoch_idx + 1:04d}{suffix}.png")
