import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision.io import write_png
from torchvision.transforms import v2
from tqdm import tqdm

from modelling.dcgan import DcGanDiscriminator, DcGanGenerator


def normalize(x: Tensor) -> Tensor:
    return (x / 255 - 0.5) / 0.5


def unnormalize(x: Tensor) -> Tensor:
    return ((x * 0.5 + 0.5) * 255).round().to(torch.uint8)


if __name__ == "__main__":
    device = "cuda"
    n_epochs = 10
    batch_size = 128

    disc = DcGanDiscriminator().to(device)
    gen = DcGanGenerator().to(device)
    print(disc)
    print(gen)

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
            loss_d = -F.logsigmoid(disc(reals)).mean() - F.logsigmoid(-disc(fakes)).mean()
            loss_d.backward()
            optim_d.step()
            optim_d.zero_grad()

            disc.requires_grad_(False)
            zs = torch.randn(batch_size, 128, device=device)
            fakes = gen(zs)
            loss_g = -F.logsigmoid(disc(fakes)).mean()
            loss_g.backward()
            optim_g.step()
            optim_g.zero_grad()
            disc.requires_grad_(True)

        with torch.no_grad():
            fakes = gen(fixed_zs)
        fakes = fakes.cpu().view(10, 10, 3, 64, 64).permute(2, 0, 3, 1, 4).reshape(3, 640, 640)
        write_png(unnormalize(fakes), f"images_celeba/epoch{epoch_idx + 1:04d}.png")
