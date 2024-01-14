import argparse
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import wandb
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets, io
from torchvision.transforms import v2
from tqdm import tqdm

from diff_augment import DiffAugment
from ema import EMA
from fid import FID
from losses import get_loss, get_regularizer
from modelling import build_discriminator, build_generator


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def unnormalize(x: Tensor) -> Tensor:
    return ((x * 0.5 + 0.5) * 255).round().clip(0, 255).to(torch.uint8)


def apply_spectral_norm(m: nn.Module):
    if isinstance(m, (nn.Linear, nn.modules.conv._ConvNd, nn.Embedding)):
        nn.utils.parametrizations.spectral_norm(m)


@dataclass
class TrainConfig:
    model: str = "dcgan"
    img_size: int = 64
    disc_kwargs: dict = field(default_factory=dict)
    gen_kwargs: dict = field(default_factory=dict)
    sn_disc: bool = False
    sn_gen: bool = False

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = False
    channels_last: bool = False
    compile: bool = False
    grad_accum: int = 1
    n_iters: int = 30_000
    n_disc: int = 1
    lr: float = 2e-4
    optimizer: str = "Adam"
    optimizer_kwargs: dict = field(default_factory=dict)
    batch_size: int = 64
    method: str = "gan"
    regularizer: str = "none"
    diff_augment: bool = False

    run_name: str = "dcgan_celeba"
    log_img_interval: int = 1_000
    fid_interval: int = 5_000


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

    disc = build_discriminator(cfg.model, img_size=cfg.img_size, **cfg.disc_kwargs).to(cfg.device)
    gen = build_generator(cfg.model, img_size=cfg.img_size, **cfg.gen_kwargs).to(cfg.device)
    disc.apply(apply_spectral_norm) if cfg.sn_disc else None
    gen.apply(apply_spectral_norm) if cfg.sn_gen else None
    disc = nn.Sequential(DiffAugment(), disc) if cfg.diff_augment else disc
    if cfg.channels_last:
        gen.to(memory_format=torch.channels_last)
        disc.to(memory_format=torch.channels_last)
    if cfg.compile:
        gen.compile()
        disc.compile()

    print(disc)
    print(gen)

    gen_ema = EMA(gen)
    criterion = get_loss(cfg.method)
    regularizer = get_regularizer(cfg.regularizer)

    optim_cls = getattr(torch.optim, cfg.optimizer)
    optim_d = optim_cls(disc.parameters(), cfg.lr, **cfg.optimizer_kwargs)
    optim_g = optim_cls(gen.parameters(), cfg.lr, **cfg.optimizer_kwargs)

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(cfg.img_size, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(cfg.img_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    ds = datasets.CelebA("data", transform=transform, download=True)
    dloader = DataLoader(ds, cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    dloader = cycle(dloader)

    autocast_ctx = torch.autocast(cfg.device, dtype=torch.bfloat16, enabled=cfg.mixed_precision)
    fixed_zs = torch.randn(100, 128, device=cfg.device)

    fid_scorer = FID(cfg.device)
    celeba_stats_path = Path(f"celeba{cfg.img_size}_stats.pth")

    if not celeba_stats_path.exists():
        celeba_stats = fid_scorer.compute_stats(lambda: next(dloader)[0].to(cfg.device))
        torch.save(celeba_stats, celeba_stats_path)

    else:
        celeba_stats = torch.load(celeba_stats_path, map_location="cpu")

    logger = wandb.init(project="dcgan_celeba", name=cfg.run_name, config=vars(cfg))
    log_img_dir = Path("images") / cfg.run_name
    log_img_dir.mkdir(parents=True, exist_ok=True)

    step = 0  # generator update step
    pbar = tqdm(total=cfg.n_iters, dynamic_ncols=True)
    while step < cfg.n_iters:
        for i in range(cfg.n_disc):
            if i == cfg.n_disc - 1:
                cached_reals = []

            if cfg.method == "wgan":
                with torch.no_grad():
                    for p in disc.parameters():
                        p.clip_(-0.01, 0.01)

            for _ in range(cfg.grad_accum):
                reals, _ = next(dloader)
                reals = reals.to(cfg.device)
                if cfg.channels_last:
                    reals = reals.to(memory_format=torch.channels_last)
                cached_reals.append(reals.clone())  # cached for generator later
                reals.requires_grad_()

                zs = torch.randn(cfg.batch_size, 128, device=cfg.device)
                with autocast_ctx:
                    with torch.no_grad():
                        fakes = gen(zs)
                    d_reals = disc(reals)
                    d_fakes = disc(fakes)
                    loss_d = criterion.d_loss(d_reals, d_fakes)
                    if regularizer is not None:
                        loss_d += regularizer(disc, reals, fakes, d_reals)
                loss_d.backward()

            optim_d.step()
            optim_d.zero_grad()

        disc.requires_grad_(False)
        for i in range(cfg.grad_accum):
            zs = torch.randn(cfg.batch_size, 128, device=cfg.device)
            with autocast_ctx:
                loss_g = criterion.g_loss(disc(gen(zs)), disc, cached_reals[i])
            loss_g.backward()
        optim_g.step()
        optim_g.zero_grad()
        disc.requires_grad_(True)
        gen_ema.step()

        step += 1
        pbar.update()
        if step % 50 == 0:
            log_dict = {
                "loss/d": loss_d.item(),
                "d/real": d_reals.detach().mean().item(),
                "d/fake": d_fakes.detach().mean().item(),
                "loss/g": loss_g.item(),
            }
            logger.log(log_dict, step=step)

        if step % cfg.log_img_interval == 0:
            for suffix, model in [("", gen), ("_ema", gen_ema)]:
                with autocast_ctx, torch.no_grad():
                    fakes = model(fixed_zs)
                fakes = fakes.cpu().unflatten(0, (10, 10)).permute(2, 0, 3, 1, 4).flatten(1, 2).flatten(-2, -1)
                fakes = unnormalize(fakes)
                io.write_png(fakes, f"{log_img_dir}/step{step // 1000:04d}k{suffix}.png")

        if step % cfg.fid_interval == 0:

            def closure():
                zs = torch.randn(cfg.batch_size, 128, device=cfg.device)
                with autocast_ctx, torch.no_grad():
                    return gen_ema(zs).float()

            stats = fid_scorer.compute_stats(closure)
            fid_score = fid_scorer.fid_score(celeba_stats, stats)
            logger.log({"fid/ema": fid_score}, step=step)
