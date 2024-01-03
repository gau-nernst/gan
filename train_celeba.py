import argparse
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets, io
from torchvision.transforms import v2
from tqdm import tqdm

import wandb
from ema import EMA
from modelling.dcgan import DcGanDiscriminator, DcGanGenerator


def unnormalize(x: Tensor) -> Tensor:
    return ((x * 0.5 + 0.5) * 255).round().to(torch.uint8)


class GAN:
    @staticmethod
    def d_loss(disc: nn.Module, reals: Tensor, fakes: Tensor) -> Tensor:
        return -F.logsigmoid(disc(reals)).mean() - F.logsigmoid(-disc(fakes)).mean()

    @staticmethod
    def g_loss(d_fakes: Tensor, disc: nn.Module, reals: Tensor) -> Tensor:
        return -F.logsigmoid(d_fakes).mean()


class WGAN:
    @staticmethod
    def d_loss(disc: nn.Module, reals: Tensor, fakes: Tensor) -> Tensor:
        with torch.no_grad():
            for p in disc.parameters():
                p.clip_(-0.01, 0.01)
        return -disc(reals).mean() + disc(fakes).mean()

    @staticmethod
    def g_loss(d_fakes: Tensor, disc: nn.Module, reals: Tensor) -> Tensor:
        return -d_fakes.mean()


class RGAN:
    @staticmethod
    def d_loss(disc: nn.Module, reals: Tensor, fakes: Tensor) -> Tensor:
        return -F.logsigmoid(disc(reals) - disc(fakes)).mean()

    @staticmethod
    def g_loss(d_fakes: Tensor, disc: nn.Module, reals: Tensor) -> Tensor:
        with torch.no_grad():
            d_reals = disc(reals)
        return -F.logsigmoid(d_fakes - d_reals).mean()


@dataclass
class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = False
    n_iters: int = 30_000
    n_disc: int = 1
    lr: float = 2e-4
    optimizer: str = "AdamW"
    optimizer_kwargs: dict = field(default_factory=dict)
    batch_size: int = 128
    method: str = "gan"
    run_name: str = "dcgan_celeba"
    log_img_interval: int = 1_000


def make_parser(fn):
    parser = argparse.ArgumentParser()

    for k, v in inspect.signature(fn).parameters.items():
        if v.annotation in (str, int, float):
            parser.add_argument(f"--{k}", type=v.annotation, default=v.default)
        elif v.annotation is bool:
            parser.add_argument(f"--{k}", action="store_true")
        elif v.annotation is dict:
            parser.add_argument(f"--{k}", type=json.loads, default=dict())
        else:
            raise RuntimeError(f"Unsupported type {v.annotation} for argument {k}")

    return parser


def cycle(iterator):
    while True:
        for x in iterator:
            yield x


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
    criterion = {
        "gan": GAN,
        "wgan": WGAN,
        "rgan": RGAN,
    }[cfg.method]

    optim_cls = getattr(torch.optim, cfg.optimizer)
    optim_d = optim_cls(disc.parameters(), cfg.lr, weight_decay=0, **cfg.optimizer_kwargs)
    optim_g = optim_cls(gen.parameters(), cfg.lr, weight_decay=0, **cfg.optimizer_kwargs)

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(64, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(64),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    ds = datasets.CelebA("data", transform=transform, download=True)
    dloader = DataLoader(ds, cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    dloader = cycle(dloader)

    autocast_ctx = torch.autocast(cfg.device, dtype=torch.bfloat16, enabled=cfg.mixed_precision)
    fixed_zs = torch.randn(100, 128, device=cfg.device)

    logger = wandb.init(project="dcgan_celeba", name=cfg.run_name, config=vars(cfg))
    log_img_dir = Path("images") / cfg.run_name
    log_img_dir.mkdir(parents=True, exist_ok=True)

    step = 0  # generator update step
    pbar = tqdm(total=cfg.n_iters)
    while step < cfg.n_iters:
        for _ in range(cfg.n_disc):
            reals, _ = next(dloader)
            reals = reals.to(cfg.device)

            zs = torch.randn(cfg.batch_size, 128, device=cfg.device)
            with autocast_ctx:
                with torch.no_grad():
                    fakes = gen(zs)
                loss_d = criterion.d_loss(disc, reals, fakes)
            loss_d.backward()
            optim_d.step()
            optim_d.zero_grad()

        disc.requires_grad_(False)
        zs = torch.randn(cfg.batch_size, 128, device=cfg.device)
        with autocast_ctx:
            loss_g = criterion.g_loss(disc(gen(zs)), disc, reals)
        loss_g.backward()
        optim_g.step()
        optim_g.zero_grad()
        disc.requires_grad_(True)
        gen_ema.step()

        step += 1
        pbar.update()
        if step % 50 == 0:
            logger.log({"loss/d": loss_d.item(), "loss/g": loss_g.item()}, step=step)

        if step % cfg.log_img_interval == 0:
            for suffix, model in [("", gen), ("_ema", gen_ema)]:
                with autocast_ctx, torch.no_grad():
                    fakes = model(fixed_zs)
                fakes = fakes.cpu().view(10, 10, 3, 64, 64).permute(2, 0, 3, 1, 4).reshape(3, 640, 640)
                fakes = unnormalize(fakes)
                io.write_png(fakes, f"{log_img_dir}/step{step // 1000:04d}k{suffix}.png")
