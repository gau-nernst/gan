import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision.io import write_png
from torchvision.transforms import v2
from tqdm import tqdm

from ema import EMA
from modelling.dcgan import DcGanDiscriminator, DcGanGenerator


def unnormalize(x: Tensor) -> Tensor:
    return ((x * 0.5 + 0.5) * 255).round().to(torch.uint8)


if __name__ == "__main__":
    device = "cuda"
    n_epochs = 10
    batch_size = 128
    method = "wgan"

    disc = DcGanDiscriminator().to(device)
    gen = DcGanGenerator().to(device)
    print(disc)
    print(gen)

    gen_ema = EMA(gen)

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

    fixed_zs = torch.randn(100, 128, device=device)
    step = 0

    for epoch_idx in range(n_epochs):
        for reals, _ in tqdm(dloader):
            reals = reals.to(device)
            zs = torch.randn(batch_size, 128, device=device)
            with torch.no_grad():
                fakes = gen(zs)
            d_reals = disc(reals)
            d_fakes = disc(fakes)
            if method == "wgan":
                with torch.no_grad():
                    for p in disc.parameters():
                        p.clip_(-0.01, 0.01)
                loss_d = -d_reals.mean() + d_fakes.mean()
            else:
                loss_d = -F.logsigmoid(d_reals).mean() - F.logsigmoid(-d_fakes).mean()
            loss_d.backward()
            optim_d.step()
            optim_d.zero_grad()

            disc.requires_grad_(False)
            zs = torch.randn(batch_size, 128, device=device)
            d_fakes = disc(gen(zs))
            if method == "wgan":
                loss_g = -d_fakes.mean()
            else:
                loss_g = -F.logsigmoid(d_fakes).mean()
            loss_g.backward()
            optim_g.step()
            optim_g.zero_grad()
            disc.requires_grad_(True)
            gen_ema.step()

        for suffix, model in [("", gen), ("_ema", gen_ema)]:
            with torch.no_grad():
                fakes = model(fixed_zs)
            fakes = fakes.cpu().view(10, 10, 3, 64, 64).permute(2, 0, 3, 1, 4).reshape(3, 640, 640)
            write_png(unnormalize(fakes), f"images_celeba/epoch{epoch_idx + 1:04d}{suffix}.png")
