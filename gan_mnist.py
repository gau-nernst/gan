from functools import partial
from typing import Callable

import einops
import torch
import torch.utils.tensorboard
import torchvision
import torchvision.transforms as TT
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm

from training import train_step


def conv_norm_act(
    in_dim: int,
    out_dim: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    conv: Callable[..., nn.Module] = nn.Conv2d,
    norm: Callable[[int], nn.Module] = nn.BatchNorm2d,
    act: Callable[[], nn.Module] = partial(nn.ReLU, inplace=True)
):
    return nn.Sequential(
        conv(in_dim, out_dim, kernel_size, stride, padding, bias=norm is None),
        norm(out_dim) if norm is not None else nn.Identity(),
        act(),
    )


def init_module(module: nn.Module, nonlinearity="relu"):
    for m in (module, *module.modules()):
        if isinstance(m, nn.modules.conv._ConvNd):
            nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity=nonlinearity)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class Discriminator(nn.Sequential):
    def __init__(self, norm=None, act=None):
        super().__init__()
        kwargs = dict(kernel_size=3, padding=1)
        if norm is not None:
            kwargs.update(norm=norm)
        if act is not None:
            kwargs.update(act=act)
        
        self.layer0 = conv_norm_act(1, 64, stride=2, **kwargs)      # (64, 14, 14)
        self.layer1 = conv_norm_act(64, 128, stride=2, **kwargs)    # (128, 7, 7)
        self.layer2 = conv_norm_act(128, 256, **kwargs)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))                    # (128, 1, 1)
        self.flatten = nn.Flatten()                                 # (128)


class Generator(nn.Module):
    def __init__(self, z_dim, act=None, norm=None):
        super().__init__()
        self.z_dim = z_dim
        kwargs = dict(kernel_size=4, stride=2, padding=1, conv=nn.ConvTranspose2d)
        if norm is not None:
            kwargs.update(norm=norm)
        if act is not None:
            kwargs.update(act=act)

        self.layers = nn.Sequential(
            conv_norm_act(z_dim // 4, 512, **kwargs),   # (512, 4, 4)
            conv_norm_act(512, 256, **kwargs),          # (256, 8, 8)
            conv_norm_act(256, 128, **kwargs),          # (128, 16, 16)
            conv_norm_act(128, 64, **kwargs),           # (64, 32, 32)
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    
    def forward(self, x):
        x = einops.rearrange(x, "b (c h w) -> b c h w", h=2, w=2)
        x = self.layers(x)
        x = x[:, :, 2:30, 2:30]
        return x


def repeat_dataloader(dataloader):
    while True:
        for data in dataloader:
            yield data


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
    G = Generator(z_dim).to(device)
    init_module(D, "relu")
    init_module(G, "relu")

    optim_d = torch.optim.Adam(D.parameters(), 1e-4)
    optim_g = torch.optim.Adam(G.parameters(), 1e-4)

    ncol, nrow = 10, 10
    fixed_noise = torch.randn(ncol * nrow, z_dim, device=device)

    tb_writer = torch.utils.tensorboard.SummaryWriter()
    tb_writer.add_graph(D, torch.randn(1, 1, 28, 28, device=device))
    tb_writer.add_graph(G, torch.randn(1, z_dim, device=device))

    def add_images(self, tag, images, *args, **kwargs):
        grid = torchvision.utils.make_grid(images, nrow=ncol, normalize=True, value_range=(-1, 1))
        self.add_image(tag, grid, *args, **kwargs)
    tb_writer.add_images = add_images.__get__(tb_writer)    # patch .add_images()

    x_reals = torch.stack([ds[i][0] for i in range(ncol * nrow)], dim=0)
    tb_writer.add_images("reals", x_reals)

    with torch.no_grad():
        tb_writer.add_images("initial fakes", G(fixed_noise))

    iter_pbar = tqdm(range(1, n_iters+1), miniters=10, dynamic_ncols=True)
    for iter_i in iter_pbar:
        x_reals, _ = next(dloader)
        x_reals = x_reals.to(device)

        loss_dict = train_step(D, G, x_reals, z_dim, optim_d, optim_g)
        iter_pbar.set_postfix(loss_dict, refresh=False)

        if iter_i % save_interval == 0:
            with torch.no_grad():
                tb_writer.add_images("fakes", G(fixed_noise), iter_i)


if __name__ == "__main__":
    main()
