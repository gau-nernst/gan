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

        img_size = config.img_size
        conv_act = ["conv", "act"]

        self.from_rgb = conv_norm_act(config.img_channels, config.min_channels, conv_act, conv1x1, act=config.act)
        in_depth = config.min_channels
        self.stages = nn.Sequential()

        while img_size > config.smallest_map_size:
            out_depth = min(in_depth * 2, config.max_channels)
            self.stages.append(DiscriminatorStage(in_depth, out_depth, config))
            in_depth = out_depth
            img_size //= 2

        last_conv = partial(nn.Conv2d, kernel_size=config.smallest_map_size)
        self.stages.append(
            nn.Sequential(
                MinibatchStdDev(),
                conv_norm_act(in_depth + 1, in_depth, conv_act, act=config.act),
                conv_norm_act(in_depth, in_depth, conv_act, last_conv, act=config.act),
                conv1x1(in_depth, 1),
            )
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)

    def grow(self):
        raise NotImplementedError

    def forward(self, imgs: Tensor):
        out = self.from_rgb(imgs)
        out = self.stages(out)
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
        self.num_stages = round(math.log2(config.img_size // config.smallest_map_size)) + 1
        init_stages = config.init_stages or self.num_stages

        self.input_norm = config.norm(config.z_dim)

        self.stages = nn.Sequential()
        self.stages.append(GeneratorStage(config.z_dim, self.get_out_channels(0), config, first_stage=True))

        for i in range(1, init_stages):
            self.stages.append(GeneratorStage(self.get_out_channels(i - 1), self.get_out_channels(i), config))

        self.to_rgb = conv1x1(self.get_out_channels(init_stages - 1), config.img_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)

    def get_out_channels(self, idx: int):
        if idx >= self.num_stages:
            raise ValueError
        channels = self.config.min_channels * self.config.img_size // self.config.smallest_map_size // 2**idx
        return min(channels, self.config.max_channels)

    def grow(self):
        n = len(self.stages)
        if n >= self.num_stages:
            raise RuntimeError("Cannot grow anymore")
        in_channels = self.get_out_channels(n - 1)
        out_channels = self.get_out_channels(n)
        self.stages.append(GeneratorStage(in_channels, out_channels, self.config))
        self.stages[-1].apply(init_new_stage)

        if out_channels != in_channels:
            new_to_rgb = conv1x1(out_channels, self.config.img_channels)
            old_weight = self.to_rgb.parametrizations.weight.original
            with torch.no_grad():
                new_to_rgb.weight[:] = F.interpolate(
                    old_weight.permute(2, 3, 0, 1),
                    new_to_rgb.weight.shape[:2],
                ).permute(2, 3, 0, 1)
            parametrize.register_parametrization(new_to_rgb, "weight", EqualizedLR(new_to_rgb.weight))
            del self.to_rgb
            self.to_rgb = new_to_rgb

    def forward(self, z_embs: Tensor):
        out = self.input_norm(z_embs[..., None, None])
        out = self.stages(out)
        out = self.to_rgb(out)
        return out


def init_weights(module: nn.Module):
    if isinstance(module, (nn.modules.conv._ConvNd, nn.Linear)):
        nn.init.normal_(module.weight)
        parametrize.register_parametrization(module, "weight", EqualizedLR(module.weight))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


@torch.no_grad()
def init_new_stage(module: nn.Module):
    if isinstance(module, nn.Conv2d):
        nn.init.zeros_(module.weight)
        ky, kx = module.weight.shape[-2:]
        module.weight[:, :, ky // 2, kx // 2] = 1.0
        parametrize.register_parametrization(module, "weight", EqualizedLR(module.weight))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
