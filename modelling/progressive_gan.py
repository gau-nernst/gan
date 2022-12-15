# Progressive GAN - https://arxiv.org/pdf/1710.10196
# See Table 2 for detailed model architecture
#
# Code reference:
# https://github.com/tkarras/progressive_growing_of_gans


import math
from functools import partial
from typing import Optional

import torch
from torch import Tensor, nn

from .base import _Act, conv_norm_act


class MinibatchStdDev(nn.Module):
    def __init__(self, group_size: int = 4):
        super().__init__()
        self.group_size = group_size

    def forward(self, imgs: Tensor):
        b, c, h, w = imgs.shape
        std = imgs.view(self.group_size, -1, c, h, w).std(dim=0, unbiased=False)
        std = std.mean([1, 2, 3], keepdim=True).repeat(self.group_size, 1, h, w)
        return torch.cat([imgs, std], dim=1)


class Discriminator(nn.Module):
    def __init__(
        self,
        img_size,
        img_depth: int = 3,
        base_depth: int = 16,
        max_depth: int = 512,
        smallest_map_size: int = 4,
        act: _Act = partial(nn.LeakyReLU, 0.2, True),
    ):
        assert img_size > 4 and math.log2(img_size).is_integer()
        super().__init__()
        kwargs = dict(norm=None, act=act)
        conv3x3 = partial(conv_norm_act, kernel_size=3, padding=1, **kwargs)
        conv1x1 = partial(conv_norm_act, kernel_size=1, **kwargs)

        self.layers = nn.Sequential()
        self.layers.append(conv1x1(img_depth, base_depth))

        while img_size > smallest_map_size:
            out_depth = min(base_depth * 2, max_depth)
            self.layers.append(conv3x3(base_depth, base_depth))
            self.layers.append(conv3x3(base_depth, out_depth, stride=2))
            base_depth = out_depth
            img_size //= 2

        self.layers.append(MinibatchStdDev())
        self.layers.append(conv3x3(base_depth + 1, base_depth))
        self.layers.append(conv_norm_act(base_depth, base_depth, smallest_map_size, **kwargs))
        self.layers.append(conv1x1(base_depth, 1))

        self.layers.apply(init_weights)

    def forward(self, imgs: Tensor, ys: Optional[Tensor] = None):
        return self.layers(imgs)


class Generator(nn.Module):
    def __init__(
        self,
        img_size: int,
        img_depth: int = 3,
        z_dim: int = 512,
        base_depth: int = 16,
        max_depth: int = 512,
        smallest_map_size: int = 4,
        act: _Act = partial(nn.LeakyReLU, 0.2, True),
    ):
        assert img_size > 4 and math.log2(img_size).is_integer()
        super().__init__()
        kwargs = dict(norm=None, act=act)
        conv3x3 = partial(conv_norm_act, kernel_size=3, padding=1, **kwargs)
        up_conv = partial(conv_norm_act, kernel_size=4, stride=2, padding=1, conv=nn.ConvTranspose2d, **kwargs)

        in_depth = z_dim
        depth = base_depth * img_size // smallest_map_size
        out_depth = min(depth, max_depth)

        self.layers = nn.Sequential()
        self.layers.append(conv_norm_act(in_depth, out_depth, smallest_map_size, **kwargs))
        self.layers.append(conv3x3(out_depth, out_depth))
        in_depth = out_depth
        depth //= 2

        while smallest_map_size < img_size:
            out_depth = min(depth, max_depth)
            self.layers.append(up_conv(in_depth, out_depth))
            self.layers.append(conv3x3(out_depth, out_depth))
            in_depth = out_depth
            depth //= 2
            smallest_map_size *= 2

        self.layers.append(nn.Conv2d(in_depth, img_depth, 3, padding=1))

        self.layers.apply(init_weights)

    def forward(self, z_embs: Tensor, ys: Optional[Tensor] = None):
        return self.layers(z_embs[:, :, None, None])


def init_weights(module: nn.Module):
    if isinstance(module, nn.modules.conv._ConvNd):
        nn.init.kaiming_normal_(module.weight)
        nn.init.zeros_(module.bias)
