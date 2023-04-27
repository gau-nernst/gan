# ESRGAN - https://arxiv.org/abs/1809.00219
#
# Code reference:
# https://github.com/xinntao/ESRGAN

import torch
from torch import Tensor, nn

from .base import _Act, conv3x3, leaky_relu


# inspired by DenseNet - https://arxiv.org/abs/1608.06993
class DenseBlock(nn.ModuleList):
    def __init__(self, channels: int, growth_rate: int = 32, n_layers: int = 5, act: _Act = leaky_relu):
        super().__init__()
        for i in range(n_layers - 1):
            self.append(nn.Sequential(conv3x3(channels + growth_rate * i, growth_rate), act()))
        self.append(conv3x3(channels + growth_rate * (n_layers - 1), channels))

    def forward(self, imgs: Tensor) -> Tensor:
        out = [imgs]
        for layer in self:
            out.append(layer(torch.cat(out, 1)))
        return out[-1]


class RRDBlock(nn.Module):
    def __init__(self, channels: int, n_blocks: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(DenseBlock(channels))

    def forward(self, imgs):
        for block in self.blocks:
            imgs = imgs + block(imgs)
        return imgs


class ESRGANGenerator(nn.Module):
    def __init__(
        self,
        img_channels: int = 3,
        base_channels: int = 64,
        n_blocks: int = 8,
        upsample: int = 2,
        act: _Act = leaky_relu,
    ):
        super().__init__()
        self.input_layer = conv3x3(img_channels, base_channels)
        self.trunk = nn.Sequential(*[RRDBlock(base_channels) for _ in range(n_blocks)])
        self.trunk.append(conv3x3(base_channels, base_channels))
        self.upsample = nn.Sequential()
        for _ in range(upsample):
            self.upsample.extend([nn.ConvTranspose2d(base_channels, base_channels, 4, 2, 1), act()])
        self.upsample.extend([conv3x3(base_channels, base_channels), act(), conv3x3(base_channels, img_channels)])

    def forward(self, imgs: Tensor) -> Tensor:
        out = self.input_layer(imgs)
        out = out + self.trunk(out)
        out = self.upsample(out)
        return out
