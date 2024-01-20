import argparse
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import wandb
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import io
from torchvision.transforms import v2
from tqdm import tqdm

from diff_augment import DiffAugment
from ema import EMA
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


# https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/
class Pix2PixDataset(Dataset):
    def __init__(self, data_dir: str):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.files = sorted(x.name for x in self.data_dir.iterdir())
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.transform(Image.open(self.data_dir / self.files[idx]).convert("RGB")).chunk(2, 2)

    def __len__(self) -> int:
        return len(self.files)


@dataclass
class TrainConfig:
    model: str = "pix2pix"
    img_size: int = 256
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
    lr: float = 2e-4
    optimizer: str = "Adam"
    optimizer_kwargs: dict = field(default_factory=dict)
    batch_size: int = 64
    method: str = "gan"
    regularizer: str = "none"
    diff_augment: bool = False

    run_name: str = "pix2pix"
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

    ds = Pix2PixDataset("/Users/thien/Downloads/facades/train")
    dloader = DataLoader(
        ds, cfg.batch_size // cfg.grad_accum, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )
    dloader = cycle(dloader)

    autocast_ctx = torch.autocast(cfg.device, dtype=torch.bfloat16, enabled=cfg.mixed_precision)
    val_ds = Pix2PixDataset("/Users/thien/Downloads/facades/val")
    fixed_As = []
    for i in range(100):
        A, B = val_ds[i]
        fixed_As.append(A)
    fixed_As = torch.stack(fixed_As, 0)

    logger = wandb.init(project="pix2pix", name=cfg.run_name, config=vars(cfg))
    log_img_dir = Path("images") / cfg.run_name
    log_img_dir.mkdir(parents=True, exist_ok=True)

    step = 0  # generator update step
    pbar = tqdm(total=cfg.n_iters, dynamic_ncols=True)
    while step < cfg.n_iters:
        for _ in range(cfg.grad_accum):
            As, Bs = next(dloader)
            As = As.to(cfg.device)
            Bs = Bs.to(cfg.device)

            if cfg.channels_last:
                As = As.to(memory_format=torch.channels_last)
                Bs = Bs.to(memory_format=torch.channels_last)

            with autocast_ctx:
                with torch.no_grad():
                    fakes = gen(As)
                d_reals = disc(As, Bs)
                d_fakes = disc(As, fakes)
                loss_d = criterion.d_loss(d_reals, d_fakes)
            loss_d.backward()
        optim_d.step()
        optim_d.zero_grad()

        disc.requires_grad_(False)
        for i in range(cfg.grad_accum):
            with autocast_ctx:
                loss_g = criterion.g_loss(disc(As, gen(As)), disc, None)
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
                    fakes = model(fixed_As)
                fakes = fakes.cpu().unflatten(0, (10, 10)).permute(2, 0, 3, 1, 4).flatten(1, 2).flatten(-2, -1)
                fakes = unnormalize(fakes)
                io.write_png(fakes, f"{log_img_dir}/step{step // 1000:04d}k{suffix}.png")
