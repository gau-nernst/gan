# Progressive GAN - https://arxiv.org/pdf/1710.10196
# See Table 2 for detailed model architecture
# Discriminator includes blurring (introduced in StyleGAN) and supports skip-connections (used in StyleGAN2)
# Not implemented features
# - Progressive growing
#
# Code reference:
# https://github.com/tkarras/progressive_growing_of_gans

import math
from functools import partial
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import parametrize

from .base import _Act, _Norm, conv1x1, conv_norm_act


class Blur(nn.Module):
    def __init__(self, kernel: List[float]):
        super().__init__()
        kernel = torch.tensor(kernel, dtype=torch.float)
        kernel = kernel.view(1, -1) * kernel.view(-1, 1)
        kernel = kernel[None, None] / kernel.sum()
        self.register_buffer("kernel", kernel)

    def forward(self, imgs: Tensor):
        channels = imgs.shape[1]
        kernel = self.kernel.expand(channels, -1, -1, -1)
        return F.conv2d(imgs, kernel, padding="same", groups=channels)


class PixelNorm(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.scale = in_dim**0.5

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x) * self.scale


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
        residual: bool = False,
        act: _Act = partial(nn.LeakyReLU, 0.2, True),
        blur_kernel: Optional[List[float]] = None,
    ):
        blur_kernel = blur_kernel or [1, 2, 1]
        super().__init__()
        conv3x3_act = partial(conv_norm_act, kernel_size=3, padding=1, order=["conv", "act"], act=act)
        self.main = nn.Sequential(
            conv3x3_act(in_dim, in_dim),
            Blur(blur_kernel),  # BlurPool from StyleGAN onwards
            conv3x3_act(in_dim, out_dim, stride=2),
        )
        self.shortcut = nn.Sequential(Blur(blur_kernel), conv1x1(in_dim, out_dim, stride=2)) if residual else None

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
        residual: bool = False,
        act: _Act = partial(nn.LeakyReLU, 0.2, True),
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
        self.layers.append(conv_act(base_depth + 1, base_depth, 3, padding=1))
        self.layers.append(conv_act(base_depth, base_depth, smallest_map_size))
        self.layers.append(conv1x1(base_depth, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)

    def forward(self, imgs: Tensor):
        return self.layers(imgs).view(-1)


class Generator(nn.Module):
    def __init__(
        self,
        img_size: int,
        img_depth: int = 3,
        z_dim: int = 512,
        base_depth: int = 16,
        max_depth: int = 512,
        smallest_map_size: int = 4,
        norm: _Norm = PixelNorm,
        act: _Act = partial(nn.LeakyReLU, 0.2, True),
        blur_kernel: List[float] = None,
    ):
        assert img_size > 4 and math.log2(img_size).is_integer()
        blur_kernel = blur_kernel or [1, 2, 1]
        super().__init__()
        conv_act_norm = partial(conv_norm_act, order=["conv", "act", "norm"], norm=norm, act=act)
        conv3x3_act_norm = partial(conv_act_norm, kernel_size=3, padding=1)

        in_depth = z_dim
        depth = base_depth * img_size // smallest_map_size
        out_depth = min(depth, max_depth)

        self.layers = nn.Sequential()
        self.layers.append(norm(in_depth))
        self.layers.append(conv_act_norm(in_depth, out_depth, smallest_map_size, conv=nn.ConvTranspose2d))
        self.layers.append(conv3x3_act_norm(out_depth, out_depth))
        in_depth = out_depth
        depth //= 2

        while smallest_map_size < img_size:
            out_depth = min(depth, max_depth)
            self.layers.append(nn.Upsample(scale_factor=2.0))
            self.layers.append(conv3x3_act_norm(in_depth, out_depth))
            self.layers.append(Blur(blur_kernel))
            self.layers.append(conv3x3_act_norm(out_depth, out_depth))
            in_depth = out_depth
            depth //= 2
            smallest_map_size *= 2

        self.layers.append(conv1x1(in_depth, img_depth))

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)

    def forward(self, z_embs: Tensor):
        return self.layers(z_embs[:, :, None, None])


class EqualizedLR(nn.Module):
    def __init__(self, weight: Tensor):
        super().__init__()
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        gain = 2**0.5  # use gain=sqrt(2) everywhere
        self.scale = gain / fan_in**0.5

    def forward(self, weight: Tensor):
        return weight * self.scale

    def extra_repr(self) -> str:
        return f"scale={self.scale}"


def init_weights(module: nn.Module):
    if isinstance(module, (nn.modules.conv._ConvNd, nn.Linear)):
        nn.init.normal_(module.weight)
        parametrize.register_parametrization(module, "weight", EqualizedLR(module.weight))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
