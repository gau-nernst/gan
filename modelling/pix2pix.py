# Code reference:
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

from functools import partial

import torch
from torch import Tensor, nn

from .base import _Act, _Norm, conv_norm_act, leaky_relu, relu


class PatchGAN(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        n_layers: int = 3,
        norm: _Norm = nn.InstanceNorm2d,
        act: _Act = leaky_relu,
    ):
        super().__init__()
        conv4x4 = partial(nn.Conv2d, kernel_size=4, padding=1)

        def get_out_c(idx: int):
            return base_channels * 2 ** min(idx, 3)

        self.append(nn.Sequential(conv4x4(in_channels, base_channels, stride=2), act()))

        for i in range(1, n_layers):
            self.append(conv_norm_act(get_out_c(i - 1), get_out_c(i), conv4x4, norm, act, stride=2))

        self.append(conv_norm_act(get_out_c(n_layers - 1), get_out_c(n_layers), conv4x4, norm, act))
        self.append(conv4x4(get_out_c(n_layers), 1))


class UnetGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_stages: int,
        base_channels: int = 64,
        norm: _Norm = nn.InstanceNorm2d,
        down_act: _Act = leaky_relu,
        up_act: _Act = relu,
    ):
        super().__init__()
        down_conv = partial(nn.Conv2d, kernel_size=4, stride=2, padding=1)
        up_conv = partial(nn.ConvTranspose2d, kernel_size=4, stride=2, padding=1)

        def get_out_c(idx: int):
            return base_channels * 2 ** min(idx, 3)

        def act_conv_norm(in_c: int, out_c: int, down: bool):
            return nn.Sequential(
                (down_act if down else up_act)(),
                (down_conv if down else up_conv)(in_c, out_c, bias=False),
                norm(out_c),
            )

        self.downs = nn.ModuleList()
        self.downs.append(down_conv(in_channels, base_channels))
        for i in range(1, n_stages - 1):
            self.downs.append(act_conv_norm(get_out_c(i - 1), get_out_c(i), True))

        self.last_stage = nn.Sequential(
            down_act(),
            down_conv(get_out_c(n_stages - 2), get_out_c(n_stages - 1)),
            up_act(),
            up_conv(get_out_c(n_stages - 1), get_out_c(n_stages - 2), bias=False),
            norm(get_out_c(n_stages - 2)),
        )

        self.ups = nn.ModuleList()
        for i in range(n_stages - 2, 0, -1):
            self.ups.append(act_conv_norm(get_out_c(i) * 2, get_out_c(i - 1), False))
        self.ups.append(nn.Sequential(up_act(), up_conv(get_out_c(0) * 2, out_channels), nn.Tanh()))

    def forward(self, imgs: Tensor):
        fmaps = []
        for down in self.downs:
            imgs = down(imgs)
            fmaps.append(imgs)

        out = self.last_stage(imgs)

        for up in self.ups:
            out = up(torch.cat([out, fmaps.pop()], 1))

        return out