import os

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


def gan_g_loss(d_fakes: Tensor):
    return -F.logsigmoid(d_fakes).mean()


def wgan_d_loss(disc: nn.Module, reals: Tensor, fakes: Tensor) -> Tensor:
    with torch.no_grad():
        for p in disc.parameters():
            p.clip_(-0.01, 0.01)
    return -disc(reals).mean() + disc(fakes).mean()


def wgan_g_loss(d_fakes: Tensor):
    return -d_fakes.mean()


if __name__ == "__main__":
    device = "cuda"
    mixed_precision = "fp16"
    n_epochs = 10
    batch_size = 128
    method = "wgan"
    log_dir = "images_celeba_wgan"

    disc = DcGanDiscriminator().to(device)
    gen = DcGanGenerator().to(device)
    print(disc)
    print(gen)

    gen_ema = EMA(gen)
    d_criterion, g_criterion = {
        "gan": (gan_d_loss, gan_g_loss),
        "wgan": (wgan_d_loss, wgan_g_loss),
    }[method]

    lr = 2e-4
    optim_d = torch.optim.AdamW(disc.parameters(), lr, betas=(0.5, 0.999), weight_decay=0)
    optim_g = torch.optim.AdamW(gen.parameters(), lr, betas=(0.5, 0.999), weight_decay=0)

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
    dloader = DataLoader(ds, batch_size, shuffle=True, num_workers=4, drop_last=True)

    autocast_ctx = torch.autocast(
        device_type=device,
        dtype=dict(none=None, fp16=torch.float16, bf16=torch.bfloat16)[mixed_precision],
        enabled=mixed_precision != "none",
    )
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision == "fp16")
    os.makedirs(log_dir, exist_ok=True)
    fixed_zs = torch.randn(100, 128, device=device)
    step = 0

    for epoch_idx in range(n_epochs):
        for reals, _ in tqdm(dloader):
            reals = reals.to(device)
            zs = torch.randn(batch_size, 128, device=device)
            with autocast_ctx:
                with torch.no_grad():
                    fakes = gen(zs)
                loss_d = d_criterion(disc, reals, fakes)
            scaler.scale(loss_d).backward()
            scaler.step(optim_d)
            optim_d.zero_grad()

            disc.requires_grad_(False)
            zs = torch.randn(batch_size, 128, device=device)
            with autocast_ctx:
                loss_g = g_criterion(disc(gen(zs)))
            scaler.scale(loss_g).backward()
            scaler.step(optim_g)
            optim_g.zero_grad()
            disc.requires_grad_(True)
            gen_ema.step()

            scaler.update()

        for suffix, model in [("", gen), ("_ema", gen_ema)]:
            with autocast_ctx, torch.no_grad():
                fakes = model(fixed_zs)
            fakes = fakes.cpu().view(10, 10, 3, 64, 64).permute(2, 0, 3, 1, 4).reshape(3, 640, 640)
            write_png(unnormalize(fakes), f"{log_dir}/epoch{epoch_idx + 1:04d}{suffix}.png")
