# Progressive GAN - https://arxiv.org/pdf/1710.10196
# See Table 2 for detailed model architecture
# Discriminator supports BlurConv (StyleGAN) and skip-connections (StyleGAN2)
# Generator has an option to use BlurConv
# Not implemented features
# - Progressive growing
#
# Code reference:
# https://github.com/tkarras/progressive_growing_of_gans

import math
from dataclasses import dataclass, replace
from functools import partial
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.utils import parametrize

from .base import _Act, _Norm, conv1x1, conv3x3, conv_norm_act
from .nvidia_ops import EqualizedLR, MinibatchStdDev, PixelNorm, blur_conv_down, up_conv_blur


@dataclass
class ProgressiveGANConfig:
    img_size: int = 128
    img_channels: int = 3
    z_dim: int = 512
    min_channels: int = 16
    max_channels: int = 512
    smallest_map_size: int = 4
    init_stages: Optional[int] = None
    grow_interval: int = 50_000
    norm: _Norm = PixelNorm
    act: _Act = partial(nn.LeakyReLU, 0.2, True)
    blur_size: Optional[int] = None
    residual_D: bool = False


class DiscriminatorStage(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        config: ProgressiveGANConfig,
    ):
        super().__init__()
        conv_act = ["conv", "act"]
        down_conv = partial(blur_conv_down, kernel_size=3, blur_size=config.blur_size)
        self.main = nn.Sequential(
            conv_norm_act(in_dim, in_dim, conv_act, act=config.act),
            conv_norm_act(in_dim, out_dim, conv_act, down_conv, act=config.act),
        )
        # skip-connection in StyleGAN2
        self.shortcut = down_conv(in_dim, out_dim, kernel_size=1) if config.residual_D else None

    def forward(self, imgs: Tensor):
        out = self.main(imgs)

        # skip-connection in StyleGAN2
        if self.shortcut is not None:
            out = (out + self.shortcut(imgs)) * 2 ** (-0.5)
        return out


class Discriminator(nn.Module):
    def __init__(self, config: Optional[ProgressiveGANConfig] = None, **kwargs):
        config = config or ProgressiveGANConfig()
        config = replace(config, **kwargs)
        assert config.img_size > 4 and math.log2(config.img_size).is_integer()
        super().__init__()
        self.config = config
        self.num_stages = int(math.log2(config.img_size // config.smallest_map_size)) + 1
        self.grow_counter = 0

        init_stages = config.init_stages or self.num_stages
        self.from_rgb = nn.Sequential(
            conv1x1(config.img_channels, self._get_in_channels(self.num_stages - init_stages)),
            config.act(),
        )
        self.prev_from_rgb = None

        self.stages = nn.Sequential()
        for i in range(self.num_stages - init_stages, self.num_stages - 1):
            self.stages.append(DiscriminatorStage(self._get_in_channels(i), self._get_in_channels(i + 1), config))

        last_channels = self._get_in_channels(self.num_stages - 1)
        self.stages.append(
            nn.Sequential(
                MinibatchStdDev(),
                conv3x3(last_channels + 1, last_channels),
                config.act(),
                nn.Conv2d(last_channels, last_channels, config.smallest_map_size),
                config.act(),
                conv1x1(last_channels, 1),
            )
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)

    def _get_in_channels(self, idx: int):
        if idx < 0 or idx >= self.num_stages:
            raise ValueError
        channels = self.config.min_channels * int(2**idx)
        return min(channels, self.config.max_channels)

    def grow(self):
        n = len(self.stages)
        if n >= self.num_stages:
            raise RuntimeError("Cannot grow anymore")

        i = self.num_stages - n
        stages = nn.Sequential(DiscriminatorStage(self._get_in_channels(i - 1), self._get_in_channels(i), self.config))
        stages.extend(self.stages)
        self.stages = stages

        self.prev_from_rgb = self.from_rgb
        self.from_rgb = nn.Sequential(
            conv1x1(self.config.img_channels, self._get_in_channels(i - 1)),
            self.config.act(),
        )

        self.grow_counter = self.config.grow_interval

    def forward(self, imgs: Tensor):
        out = self.from_rgb(imgs)
        out = self.stages[0](out)

        if self.grow_counter > 0:
            # F.interpolate() is 20% faster than F.avg_pool() on RTX 3090
            prev_out = F.interpolate(imgs, scale_factor=0.5, mode="bilinear")
            prev_out = self.prev_from_rgb(prev_out)
            out = torch.lerp(prev_out, out, self.grow_counter / self.config.grow_interval)
            self.grow_counter -= 1

        for stage in self.stages[1:]:
            out = stage(out)
        return out.view(-1)


class GeneratorStage(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: ProgressiveGANConfig,
        first_stage: bool = False,
    ):
        if first_stage:
            up_conv = partial(nn.ConvTranspose2d, kernel_size=config.smallest_map_size)
        else:
            up_conv = partial(up_conv_blur, kernel_size=3, blur_size=config.blur_size)
        order = ["conv", "act", "norm"]
        super().__init__(
            conv_norm_act(in_channels, out_channels, order, up_conv, config.norm, config.act),
            conv_norm_act(out_channels, out_channels, order, conv3x3, config.norm, config.act),
        )


class Generator(nn.Module):
    def __init__(self, config: Optional[ProgressiveGANConfig] = None, **kwargs):
        config = config or ProgressiveGANConfig()
        config = replace(config, **kwargs)
        assert config.img_size > 4 and math.log2(config.img_size).is_integer()
        super().__init__()
        self.config = config
        self.num_stages = int(math.log2(config.img_size // config.smallest_map_size)) + 1
        self.grow_counter = 0

        self.input_norm = config.norm(config.z_dim)

        self.stages = nn.Sequential()
        self.stages.append(GeneratorStage(config.z_dim, self._get_out_channels(0), config, first_stage=True))

        init_stages = config.init_stages or self.num_stages
        for i in range(1, init_stages):
            self.stages.append(GeneratorStage(self._get_out_channels(i - 1), self._get_out_channels(i), config))

        self.to_rgb = conv1x1(self._get_out_channels(init_stages - 1), config.img_channels)
        self.prev_to_rgb = None

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)

    def _get_out_channels(self, idx: int):
        if idx < 0 or idx >= self.num_stages:
            raise ValueError
        channels = self.config.min_channels * self.config.img_size // self.config.smallest_map_size // 2**idx
        return min(channels, self.config.max_channels)

    def grow(self):
        n = len(self.stages)
        if n >= self.num_stages:
            raise RuntimeError("Cannot grow anymore")

        self.stages.append(GeneratorStage(self._get_out_channels(n - 1), self._get_out_channels(n), self.config))
        self.stages[-1].apply(init_weights)

        self.prev_to_rgb = self.to_rgb
        self.to_rgb = conv1x1(self._get_out_channels(n), self.config.img_channels)
        self.to_rgb.apply(init_weights)

        self.grow_counter = self.config.grow_interval

    def forward(self, z_embs: Tensor):
        out = self.input_norm(z_embs[..., None, None])
        for stage in self.stages:
            prev_out, out = out, stage(out)
        out = self.to_rgb(out)

        if self.grow_counter > 0:
            prev_out = self.prev_to_rgb(prev_out)
            prev_out = F.interpolate(prev_out, scale_factor=2, mode="nearest")
            out = torch.lerp(prev_out, out, self.grow_counter / self.config.grow_interval)
            self.grow_counter -= 1

        return out


def init_weights(module: nn.Module):
    if isinstance(module, (nn.modules.conv._ConvNd, nn.Linear)):
        nn.init.normal_(module.weight)
        parametrize.register_parametrization(module, "weight", EqualizedLR(module.weight))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
