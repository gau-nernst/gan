# Progressive GAN - https://arxiv.org/pdf/1710.10196
# See Table 2 for detailed model architecture
# Discriminator supports BlurPool (StyleGAN) and skip-connections (StyleGAN2)
# Generator has an option to use BlurConv
# Not implemented features
# - Progressive growing
#
# Code reference:
# https://github.com/tkarras/progressive_growing_of_gans

import math
from functools import partial
from typing import List, Optional, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import parametrize

from .base import _Act, _Norm, conv1x1, conv3x3, conv_norm_act


# this is very slow
class Blur(nn.Module):
    def __init__(self, kernel: Optional[List[float]] = None, stride=1):
        kernel = kernel or [1, 2, 1]
        super().__init__()
        k = len(kernel)
        kernel = torch.tensor(kernel, dtype=torch.float)
        kernel = kernel.view(1, -1) * kernel.view(-1, 1)
        kernel = kernel.view(1, 1, k, k) / kernel.sum()
        self.register_buffer("kernel", kernel)
        self.stride = stride
        self.padding = (k - 1) // 2

    def forward(self, imgs: Tensor):
        n, c, h, w = imgs.shape
        imgs = imgs.view(n * c, 1, h, w)
        imgs = F.conv2d(imgs, self.kernel, stride=self.stride, padding=self.padding)
        imgs = imgs.view(n, c, h, w)
        return imgs


def resample_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    padding: int = 0,
    resample: Literal["up", "down"] = "down",
    use_blur: bool = False,
    blur_kernel: Optional[List[float]] = None,
):
    conv = partial(nn.Conv2d, in_channels, out_channels, kernel_size, padding=padding)
    up = partial(nn.Upsample, scale_factor=2.0)
    blur = partial(Blur, blur_kernel=blur_kernel)

    if use_blur:
        if resample == "up":
            layers = [conv(), up(), blur()] if kernel_size == 1 else [up(), conv(), blur()]
        else:
            layers = [blur(stride=2), conv()] if kernel_size == 1 else [blur(), conv(stride=2)]

    else:
        layers = [up(), conv()] if resample == "up" else [conv(stride=2)]

    return nn.Sequential(*layers)


class PixelNorm(nn.Module):
    def __init__(self, in_dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = in_dim**0.5
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, eps=self.eps) * self.scale


class MinibatchStdDev(nn.Module):
    def __init__(self, group_size: int = 4):
        super().__init__()
        self.group_size = group_size

    def forward(self, imgs: Tensor):
        _, c, h, w = imgs.shape
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
        use_blur: bool = False,
        blur_kernel: Optional[List[float]] = None,
    ):
        super().__init__()
        conv_act = partial(conv_norm_act, order=["conv", "act"], conv=conv3x3, act=act)
        down_conv = partial(resample_conv, resample="down", use_blur=use_blur, blur_kernel=blur_kernel)

        self.main = nn.Sequential(
            conv_act(in_dim, in_dim),
            conv_act(in_dim, out_dim, conv=partial(down_conv, kernel_size=3, padding=1)),
        )
        self.shortcut = down_conv(in_dim, out_dim, 1) if residual else None  # skip-connection in StyleGAN2

    def forward(self, imgs: Tensor):
        out = self.main(imgs)
        if self.shortcut is not None:  # skip-connection in StyleGAN2
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
        use_blur: bool = False,
        blur_kernel: Optional[List[float]] = None,
    ):
        assert img_size > 4 and math.log2(img_size).is_integer()
        super().__init__()
        conv_act = partial(conv_norm_act, order=["conv", "act"], conv=conv3x3, act=act)
        stage = partial(DiscriminatorStage, act=act, residual=residual, use_blur=use_blur, blur_kernel=blur_kernel)

        self.layers = nn.Sequential()
        self.layers.append(conv_act(img_depth, base_depth, conv=conv1x1))

        while img_size > smallest_map_size:
            out_depth = min(base_depth * 2, max_depth)
            self.layers.append(stage(base_depth, out_depth))
            base_depth = out_depth
            img_size //= 2

        self.layers.append(MinibatchStdDev())
        self.layers.append(conv_act(base_depth + 1, base_depth))
        self.layers.append(conv_act(base_depth, base_depth, conv=partial(nn.Conv2d, kernel_size=smallest_map_size)))
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
        use_blur: bool = False,
        blur_kernel: Optional[List[float]] = None,
    ):
        assert img_size > 4 and math.log2(img_size).is_integer()
        super().__init__()
        up_conv3x3 = partial(
            resample_conv, kernel_size=3, padding=1, resample="up", use_blur=use_blur, blur_kernel=blur_kernel
        )
        conv_act_norm = partial(conv_norm_act, order=["conv", "act", "norm"], conv=conv3x3, act=act, norm=norm)

        in_depth = z_dim
        depth = base_depth * img_size // smallest_map_size
        out_depth = min(depth, max_depth)

        self.layers = nn.Sequential()
        self.layers.append(norm(in_depth))
        self.layers.append(
            conv_act_norm(in_depth, out_depth, conv=partial(nn.ConvTranspose2d, kernel_size=smallest_map_size))
        )
        self.layers.append(conv_act_norm(out_depth, out_depth))
        in_depth = out_depth
        depth //= 2

        while smallest_map_size < img_size:
            out_depth = min(depth, max_depth)
            self.layers.append(conv_act_norm(in_depth, out_depth, conv=up_conv3x3))
            self.layers.append(conv_act_norm(out_depth, out_depth))
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
