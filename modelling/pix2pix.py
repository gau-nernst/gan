# pix2pix - https://arxiv.org/abs/1611.07004
#
# Code reference:
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix


import torch
from torch import Tensor, nn


class PatchGan(nn.Sequential):
    def __init__(self, A_channels: int = 3, B_channels: int = 3, base_dim: int = 64, depth: int = 3):
        super().__init__()
        self.append(nn.Conv2d(A_channels + B_channels, base_dim, 4, 2, 1))
        self.append(nn.LeakyReLU(0.2, inplace=True))
        in_ch = base_dim

        for i in range(1, depth):
            out_ch = base_dim * 2 ** min(i, 3)
            self.append(nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False))
            self.append(nn.InstanceNorm2d(out_ch))
            self.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = out_ch

        out_ch = base_dim * 2 ** min(depth, 3)
        self.append(nn.Conv2d(in_ch, out_ch, 4, 1, 1, bias=False))
        self.append(nn.InstanceNorm2d(out_ch))
        self.append(nn.LeakyReLU(0.2, inplace=True))
        self.append(nn.Conv2d(out_ch, 1, 4, 1, 1))

    def forward(self, imgs_A: Tensor, imgs_B: Tensor):
        return super().forward(torch.cat([imgs_A, imgs_B], dim=1))


class UnetGenerator(nn.Module):
    def __init__(self, A_channels: int = 3, B_channels: int = 3, depth: int = 7, base_dim: int = 64):
        super().__init__()
        self.downs = nn.ModuleList()
        self.downs.append(nn.Conv2d(A_channels, base_dim, 4, 2, 1))
        in_ch = base_dim
        for i in range(1, depth - 1):
            out_ch = base_dim * 2 ** min(i, 3)
            block = nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(out_ch),
            )
            self.downs.append(block)
            in_ch = out_ch

        out_ch = base_dim * 2 ** min(depth - 1, 3)
        self.last_stage = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_ch, out_ch, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_ch, in_ch, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(in_ch),
        )

        self.ups = nn.ModuleList()
        for i in range(depth - 2, 0, -1):
            out_ch = base_dim * 2 ** min(i, 3)
            block = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(out_ch),
            )
            in_ch = out_ch
        self.ups.append(nn.Sequential(nn.ReLU(inplace=True), nn.ConvTranspose2d(in_ch, B_channels), nn.Tanh()))

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)

    def forward(self, imgs: Tensor):
        fmaps = []
        for down in self.downs:
            imgs = down(imgs)
            fmaps.append(imgs)

        out = self.last_stage(imgs)

        for up in self.ups:
            out = up(torch.cat([out, fmaps.pop()], 1))

        return out


def init_weights(module: nn.Module):
    if isinstance(module, (nn.modules.conv._ConvNd, nn.modules.conv._ConvTransposeNd)):
        nn.init.normal_(module.weight, 0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.modules.InstanceNorm2d):
        nn.init.normal_(module.weight, 1, 0.02)
        nn.init.zeros_(module.bias)
