# Progressive GAN - https://arxiv.org/pdf/1710.10196
# See Table 2 for detailed model architecture
# Discriminator includes blurring (introduced in StyleGAN) and supports skip-connections (used in StyleGAN2)
# Not implemented features
# - Progressive growing and Equalized learning rate
#
# Code reference:
# https://github.com/tkarras/progressive_growing_of_gans

import math
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import Blur, PixelNorm, _Act, _Norm, conv_norm_act


class MinibatchStdDev(nn.Module):
    def __init__(self, group_size: int = 4):
        super().__init__()
        self.group_size = group_size

    def forward(self, imgs: Tensor):
        b, c, h, w = imgs.shape
        std = imgs.view(self.group_size, -1, c, h, w).std(dim=0, unbiased=False)
        std = std.mean([1, 2, 3], keepdim=True).repeat(self.group_size, 1, h, w)
        return torch.cat([imgs, std], dim=1)


class DiscriminatorStage(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        act: _Act = partial(nn.LeakyReLU, 0.2, True),
        residual: bool = False,
    ):
        super().__init__()
        conv3x3_act = partial(conv_norm_act, kernel_size=3, padding=1, order=["conv", "act"], act=act)
        self.main = nn.Sequential(
            conv3x3_act(in_dim, in_dim),
            Blur(),  # BlurPool from StyleGAN onwards
            conv3x3_act(in_dim, out_dim, stride=2),
        )
        self.shortcut = nn.Sequential(Blur(), nn.Conv2d(in_dim, out_dim, 1, 2)) if residual else None

    def forward(self, imgs: Tensor):
        out = self.main(imgs)
        if self.shortcut is not None:  # for StyleGAN2
            out = (out + self.shortcut(imgs)) * 2 ** (-0.5)
        return out


class Discriminator(nn.Module):
    def __init__(
        self,
        img_size,
        img_depth: int = 3,
        base_depth: int = 16,
        max_depth: int = 512,
        smallest_map_size: int = 4,
        act: _Act = partial(nn.LeakyReLU, 0.2, True),
        residual: bool = False,
    ):
        assert img_size > 4 and math.log2(img_size).is_integer()
        super().__init__()
        conv_act = partial(conv_norm_act, order=["conv", "act"], act=act)
        stage = partial(DiscriminatorStage, act=act, residual=residual)

        self.layers = nn.Sequential()
        self.layers.append(conv_act(img_depth, base_depth, 1))

        while img_size > smallest_map_size:
            out_depth = min(base_depth * 2, max_depth)
            self.layers.append(stage(base_depth, out_depth))
            base_depth = out_depth
            img_size //= 2

        self.layers.append(MinibatchStdDev())
        self.layers.append(conv_act(base_depth + 1, base_depth, 3, 1, 1))
        self.layers.append(conv_act(base_depth, base_depth, smallest_map_size))
        self.layers.append(conv_act(base_depth, 1, 1))

        self.layers.apply(init_weights)

    def forward(self, imgs: Tensor):
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
        norm: Optional[_Norm] = PixelNorm,
        act: _Act = partial(nn.LeakyReLU, 0.2, True),
    ):
        assert img_size > 4 and math.log2(img_size).is_integer()
        super().__init__()
        conv_act_norm = partial(conv_norm_act, order=["conv", "act", "norm"], norm=norm, act=act)

        in_depth = z_dim
        depth = base_depth * img_size // smallest_map_size
        out_depth = min(depth, max_depth)

        self.layers = nn.Sequential()
        self.layers.append(conv_act_norm(in_depth, out_depth, smallest_map_size, conv=nn.ConvTranspose2d))
        self.layers.append(conv_act_norm(out_depth, out_depth, 3, padding=1))
        in_depth = out_depth
        depth //= 2

        while smallest_map_size < img_size:
            out_depth = min(depth, max_depth)
            self.layers.append(nn.Upsample(scale_factor=2.0))
            self.layers.append(conv_act_norm(in_depth, out_depth, 3, padding=1))
            self.layers.append(Blur())
            self.layers.append(conv_act_norm(out_depth, out_depth, 3, padding=1))
            in_depth = out_depth
            depth //= 2
            smallest_map_size *= 2

        self.layers.append(nn.Conv2d(in_depth, img_depth, 3, padding=1))

        self.layers.apply(init_weights)

    def forward(self, z_embs: Tensor):
        return self.layers(F.normalize(z_embs)[:, :, None, None])


def init_weights(module: nn.Module):
    if isinstance(module, (nn.modules.conv._ConvNd, nn.Linear)):
        nn.init.kaiming_normal_(module.weight)
        nn.init.zeros_(module.bias)
