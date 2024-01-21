# pix2pix - https://arxiv.org/abs/1611.07004
#
# Code reference:
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix


import torch
from torch import Tensor, nn


class PatchGan(nn.Sequential):
    def __init__(self, A_channels: int = 3, B_channels: int = 3, base_dim: int = 64, depth: int = 3) -> None:
        super().__init__()
        self.append(nn.Conv2d(A_channels + B_channels, base_dim, 4, 2, 1))
        self.append(nn.LeakyReLU(0.2, inplace=True))
        in_ch = base_dim

        for i in range(1, depth):
            out_ch = base_dim * 2 ** min(i, 3)
            self.append(nn.Conv2d(in_ch, out_ch, 4, 2, 1))
            self.append(nn.InstanceNorm2d(out_ch))
            self.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = out_ch

        out_ch = base_dim * 2 ** min(depth, 3)
        self.append(nn.Conv2d(in_ch, out_ch, 3, 1, 1))  # original code uses kernel_size=4 here
        self.append(nn.InstanceNorm2d(out_ch))
        self.append(nn.LeakyReLU(0.2, inplace=True))
        self.append(nn.Conv2d(out_ch, 1, 3, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)

    def forward(self, As: Tensor, Bs: Tensor):
        return super().forward(torch.cat([As, Bs], dim=1))


class UnetGenerator(nn.Module):
    def __init__(self, A_channels: int = 3, B_channels: int = 3, depth: int = 7, base_dim: int = 64):
        super().__init__()
        self.down_blocks = nn.ModuleList()
        self.down_blocks.append(nn.Conv2d(A_channels, base_dim, 4, 2, 1))
        in_ch = base_dim
        for i in range(1, depth - 1):
            out_ch = base_dim * 2 ** min(i, 3)
            block = nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_ch, out_ch, 4, 2, 1),
                nn.InstanceNorm2d(out_ch),
            )
            self.down_blocks.append(block)
            in_ch = out_ch

        out_ch = base_dim * 2 ** min(depth - 1, 3)
        self.inner_most = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_ch, out_ch, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_ch, in_ch, 4, 2, 1),
            nn.InstanceNorm2d(in_ch),
        )

        self.up_blocks = nn.ModuleList()
        for i in range(depth - 3, -1, -1):
            out_ch = base_dim * 2 ** min(i, 3)
            block = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_ch * 2, out_ch, 4, 2, 1),
                nn.InstanceNorm2d(out_ch),
            )
            self.up_blocks.append(block)
            in_ch = out_ch
        self.up_blocks.append(
            nn.Sequential(
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_ch * 2, B_channels, 4, 2, 1),
                nn.Tanh(),
            )
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)

    def forward(self, x: Tensor):
        fmaps = [x]
        for down in self.down_blocks:
            fmaps.append(down(fmaps[-1]))

        out = self.inner_most(fmaps[-1])

        for up in self.up_blocks:
            out = up(torch.cat([out, fmaps.pop()], 1))
        return out


def init_weights(module: nn.Module):
    if isinstance(module, (nn.modules.conv._ConvNd, nn.modules.conv._ConvTransposeNd)):
        nn.init.normal_(module.weight, 0, 0.02)
        nn.init.zeros_(module.bias)
