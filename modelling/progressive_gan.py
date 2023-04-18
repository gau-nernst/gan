# Progressive GAN - https://arxiv.org/pdf/1710.10196
# See Table 2 for detailed model architecture
# Discriminator supports BlurConv (StyleGAN) and skip-connections (StyleGAN2)
# Generator has an option to use BlurConv
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
    init_map_size: int = 4
    progressive_growing: bool = False
    fade_duration: int = 50_000
    norm: _Norm = PixelNorm
    act: _Act = partial(nn.LeakyReLU, 0.2, True)
    blur_size: Optional[int] = None
    residual_D: bool = False


class BaseProgressiveGAN(nn.Module):
    def __init__(self, config: Optional[ProgressiveGANConfig] = None, **kwargs):
        config = config or ProgressiveGANConfig()
        config = replace(config, **kwargs)
        assert config.img_size > 4 and math.log2(config.img_size).is_integer()
        super().__init__()
        self.config = config
        self.total_stages = int(math.log2(config.img_size // config.init_map_size)) + 1

        self.register_buffer("fade_alpha", torch.tensor(1.0))
        self.register_buffer("current_stage", torch.tensor(1 if config.progressive_growing else self.total_stages))
        self.fade_alpha: Tensor
        self.current_stage: Tensor

    def reset_parameters(self):
        self.apply(init_weights)

    def step(self):
        self.fade_alpha.add_(1 / self.config.fade_duration).clamp_(0.0, 1.0)

    def grow(self):
        if self.current_stage >= self.total_stages:
            raise RuntimeError("Cannot grow anymore")
        self.current_stage += 1
        self.fade_alpha.fill_(0.0)


class DiscriminatorStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: ProgressiveGANConfig,
        last_stage: bool = False,
    ):
        super().__init__()
        if last_stage:
            self.main = nn.Sequential(
                MinibatchStdDev(),
                conv3x3(in_channels + 1, in_channels),
                config.act(),
                nn.Conv2d(in_channels, out_channels, config.init_map_size),
                config.act(),
                conv1x1(out_channels, 1),
            )
        else:
            self.main = nn.Sequential(
                conv3x3(in_channels, in_channels),
                config.act(),
                blur_conv_down(in_channels, out_channels, 3, config.blur_size),
                config.act(),
            )

        # skip-connection in StyleGAN2
        if config.residual_D and not last_stage:
            self.shortcut = blur_conv_down(in_channels, out_channels, 1, config.blur_size)
        else:
            self.shortcut = None

    def forward(self, imgs: Tensor):
        out = self.main(imgs)

        # skip-connection in StyleGAN2
        if self.shortcut is not None:
            out = (out + self.shortcut(imgs)) * 2 ** (-0.5)
        return out


class Discriminator(BaseProgressiveGAN):
    def __init__(self, config: Optional[ProgressiveGANConfig] = None, **kwargs):
        super().__init__(config, **kwargs)
        config = self.config
        self.from_rgb = nn.ModuleList()
        self.stages = nn.ModuleList()

        in_channels = config.min_channels
        for i in range(self.total_stages):
            out_channels = min(config.min_channels * 2 ** (i + 1), config.max_channels)
            self.from_rgb.append(nn.Sequential(conv1x1(config.img_channels, in_channels), config.act()))
            self.stages.append(DiscriminatorStage(in_channels, out_channels, config, i == self.total_stages - 1))
            in_channels = out_channels

        self.reset_parameters()

    def forward(self, imgs: Tensor):
        curr_stage = self.total_stages - self.current_stage.item()

        out = self.from_rgb[curr_stage](imgs)
        out = self.stages[curr_stage](out)

        if self.fade_alpha < 1.0:
            # F.interpolate() is 20% faster than F.avg_pool() on RTX 3090
            prev_out = F.interpolate(imgs, scale_factor=0.5, mode="bilinear")
            prev_out = self.from_rgb[curr_stage + 1](prev_out)
            out = prev_out.lerp(out, self.fade_alpha)

        for stage in self.stages[curr_stage + 1 :]:
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
            up_conv = partial(nn.ConvTranspose2d, kernel_size=config.init_map_size)
        else:
            up_conv = partial(up_conv_blur, kernel_size=3, blur_size=config.blur_size)
        order = ["conv", "act", "norm"]
        super().__init__(
            conv_norm_act(in_channels, out_channels, order, up_conv, config.norm, config.act),
            conv_norm_act(out_channels, out_channels, order, conv3x3, config.norm, config.act),
        )


class Generator(BaseProgressiveGAN):
    def __init__(self, config: Optional[ProgressiveGANConfig] = None, **kwargs):
        super().__init__(config, **kwargs)
        config = self.config
        self.input_norm = config.norm(config.z_dim)
        self.stages = nn.ModuleList()
        self.to_rgb = nn.ModuleList()

        in_channels = config.z_dim
        first_out_channels = self.config.min_channels * self.config.img_size // self.config.init_map_size
        for i in range(self.total_stages):
            out_channels = min(first_out_channels // 2**i, config.max_channels)
            self.stages.append(GeneratorStage(in_channels, out_channels, config, i == 0))
            self.to_rgb.append(conv1x1(out_channels, config.img_channels))
            in_channels = out_channels

        self.reset_parameters()

    def forward(self, z_embs: Tensor):
        curr_stage = self.current_stage.item()

        out = self.input_norm(z_embs[..., None, None])
        for stage in self.stages[:curr_stage]:
            prev_out, out = out, stage(out)
        out = self.to_rgb[curr_stage - 1](out)

        if self.fade_alpha < 1.0:
            prev_out = self.to_rgb[curr_stage - 2](prev_out)
            prev_out = F.interpolate(prev_out, scale_factor=2, mode="nearest")
            out = prev_out.lerp(out, self.fade_alpha)

        return out


def init_weights(module: nn.Module):
    if isinstance(module, (nn.modules.conv._ConvNd, nn.Linear)):
        nn.init.normal_(module.weight)
        parametrize.register_parametrization(module, "weight", EqualizedLR(module.weight))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
