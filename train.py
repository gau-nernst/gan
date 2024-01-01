import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST
from tqdm import tqdm

from modelling.dcgan import DCGANDiscriminator, DCGANGenerator


if __name__ == "__main__":
    device = "cuda"

    disc = DCGANDiscriminator(img_channels=1, img_size=32).to(device)
    gen = DCGANGenerator(img_channels=1, img_size=32).to(device)

    lr = 2e-4
    optim_d = torch.optim.AdamW(disc.parameters(), lr, betas=(0.5, 0.95), weight_decay=0)
    optim_g = torch.optim.AdamW(gen.parameters(), lr, betas=(0.5, 0.95), weight_decay=0)

    images = MNIST("data", download=True).data  # (60_000, 28, 28)
    images = F.pad(images, (2, 2, 2, 2))  # (60_000, 32, 32)
    images = images / 255  # uint8 -> float
    images = (images - 0.5) / 0.5  # [0,1] -> [-1,1]
    images = images.to(device)

    n_epochs = 10
    batch_size = 64

    fixed_zs = torch.randn(100, 128, device=device)
    step = 0

    for _ in range(n_epochs):
        indices = torch.randperm(images.shape[0], device=device)
        images = images[indices]

        for i in tqdm(range(0, len(images) - batch_size + 1, batch_size)):
            reals = images[i : i + batch_size].unsqueeze(1)

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
        fakes.view(10, 10, 32, 32).permute(0, 2, 1, 3)
