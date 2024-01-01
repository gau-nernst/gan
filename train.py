from modelling.dcgan import DCGANDiscriminator, DCGANGenerator
from torchvision.datasets import MNIST
import torch.nn.functional as F
import torch


if __name__ == "__main__":
    device = "cuda"

    disc = DCGANDiscriminator().to(device)
    gen = DCGANGenerator().to(device)

    lr = 2e-4
    d_optim = torch.optim.AdamW(disc.parameters(), lr, betas=(0.5, 0.95), weight_decay=0)
    g_optim = torch.optim.AdamW(gen.parameters(), lr, betas=(0.5, 0.95), weight_decay=0)

    images = MNIST("data", download=True).data  # (60_000, 28, 28)
    images = F.pad(images, (2, 2, 2, 2))  # (60_000, 32, 32)
    images = images / 255  # uint8 -> float
    images = (images - 0.5) / 0.5  # [0,1] -> [-1,1]
    images = images.to(device)

    n_epochs = 10
    batch_size = 64

    for _ in range(n_epochs):
        indices = torch.randperm(images.shape[0], device=device)
        images = images[indices]

        for i in range(0, len(images) - batch_size + 1, batch_size):
            batch = images[i : i + batch_size]
