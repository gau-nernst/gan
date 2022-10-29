from functools import partial

import einops
import torch
import torchvision.transforms as TT
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm

from modelling import conv_norm_act, init_module
from training import gan_train_step
from utils import TensorboardWriter, repeat_dataloader


class Discriminator(nn.Sequential):
    def __init__(self, norm=nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        make_layer = partial(conv_norm_act, kernel_size=3, padding=1, norm=norm, act=act)
        
        self.layer0 = make_layer(1, 64, stride=2)   # (64, 14, 14)
        self.layer1 = make_layer(64, 128, stride=2) # (128, 7, 7)
        self.layer2 = make_layer(128, 256)          # (256, 7, 7)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))    # (256, 1, 1)
        self.flatten = nn.Flatten()                 # (128)


class Generator(nn.Module):
    def __init__(self, z_dim, norm=nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        self.z_dim = z_dim
        make_layer = partial(conv_norm_act, kernel_size=4, stride=2, padding=1, conv=nn.ConvTranspose2d, norm=norm, act=act)

        self.layers = nn.Sequential(
            make_layer(z_dim // 4, 512),    # (512, 4, 4)
            make_layer(512, 256),           # (256, 8, 8)
            make_layer(256, 128),           # (128, 16, 16)
            make_layer(128, 64),            # (64, 32, 32)
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = einops.rearrange(x, "b (c h w) -> b c h w", h=2, w=2)
        return self.layers(x)[:, :, 2:30, 2:30]


def main():
    z_dim = 128
    bsize = 32
    device = "cuda:1"
    n_iters = 20000
    save_interval = n_iters // 10

    transform = TT.Compose([
        TT.ToTensor(),
        TT.Normalize(0.5, 0.5)
    ])
    ds = MNIST("data", transform=transform)
    dloader = DataLoader(ds, batch_size=bsize, shuffle=True, num_workers=8)
    dloader = repeat_dataloader(dloader)

    D = Discriminator().to(device)
    init_module(D, "relu")
    optim_d = torch.optim.Adam(D.parameters(), 1e-4)

    G = Generator(z_dim).to(device)
    init_module(G, "relu")
    optim_g = torch.optim.Adam(G.parameters(), 1e-4)

    ncol, nrow = 10, 10
    fixed_noise = torch.randn(ncol * nrow, z_dim, device=device)
    tb_writer = TensorboardWriter(comment="mnist")

    x_reals = torch.stack([ds[i][0] for i in range(ncol * nrow)], dim=0)
    tb_writer.add_images("reals", x_reals, nrow=ncol)

    with torch.no_grad():
        tb_writer.add_images("initial fakes", G(fixed_noise), nrow=ncol)

    iter_pbar = tqdm(range(1, n_iters+1), miniters=10, dynamic_ncols=True)
    for iter_i in iter_pbar:
        x_reals, _ = next(dloader)
        x_reals = x_reals.to(device)

        loss_dict = gan_train_step(D, G, x_reals, z_dim, optim_d, optim_g)
        iter_pbar.set_postfix(loss_dict, refresh=False)
        tb_writer.add_scalars("loss", loss_dict, iter_i)

        if iter_i % save_interval == 0:
            with torch.no_grad():
                tb_writer.add_images("fakes", G(fixed_noise), iter_i, nrow=ncol)


if __name__ == "__main__":
    main()
