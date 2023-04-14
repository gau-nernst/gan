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

from torch import Tensor, nn
from torch.nn.utils import parametrize

from .base import _Act, _Norm, conv1x1, conv3x3, conv_norm_act
from .nvidia_ops import EqualizedLR, MinibatchStdDev, PixelNorm, blur_conv_down, up_conv_blur


@dataclass
class ProgressiveGANConfig:
    img_size: int = 128
    img_depth: int = 3
    z_dim: int = 512
    base_depth: int = 16
    max_depth: int = 512
    smallest_map_size: int = 4
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

        base_depth = config.base_depth
        img_size = config.img_size
        conv_act = ["conv", "act"]

        self.from_rgb = conv_norm_act(config.img_depth, base_depth, conv_act, conv1x1, act=config.act)
        self.stages = nn.Sequential()

        while img_size > config.smallest_map_size:
            out_depth = min(base_depth * 2, config.max_depth)
            self.stages.append(DiscriminatorStage(base_depth, out_depth, config))
            base_depth = out_depth
            img_size //= 2

        last_conv = partial(nn.Conv2d, kernel_size=config.smallest_map_size)
        self.stages.append(
            nn.Sequential(
                MinibatchStdDev(),
                conv_norm_act(base_depth + 1, base_depth, conv_act, act=config.act),
                conv_norm_act(base_depth, base_depth, conv_act, last_conv, act=config.act),
                conv1x1(base_depth, 1),
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
        in_depth: int,
        out_depth: int,
        config: ProgressiveGANConfig,
        first_stage: bool = False,
    ):
        if first_stage:
            up_conv = partial(nn.ConvTranspose2d, kernel_size=config.smallest_map_size)
        else:
            up_conv = partial(up_conv_blur, kernel_size=3, blur_size=config.blur_size)
        order = ["conv", "act", "norm"]
        super().__init__(
            conv_norm_act(in_depth, out_depth, order, up_conv, config.norm, config.act),
            conv_norm_act(out_depth, out_depth, order, conv3x3, config.norm, config.act),
        )


class Generator(nn.Module):
    def __init__(self, config: Optional[ProgressiveGANConfig] = None, **kwargs):
        config = config or ProgressiveGANConfig()
        config = replace(config, **kwargs)
        assert config.img_size > 4 and math.log2(config.img_size).is_integer()
        super().__init__()
        self.config = config

        in_depth = config.z_dim
        smallest_map_size = config.smallest_map_size
        depth = config.base_depth * config.img_size // smallest_map_size
        out_depth = min(depth, config.max_depth)

        self.input_norm = config.norm(in_depth)
        self.stages = nn.Sequential()
        self.stages.append(GeneratorStage(in_depth, out_depth, config, True))
        in_depth = out_depth
        depth //= 2

        while smallest_map_size < config.img_size:
            out_depth = min(depth, config.max_depth)
            self.stages.append(GeneratorStage(in_depth, out_depth, config))
            in_depth = out_depth
            depth //= 2
            smallest_map_size *= 2

        self.to_rgb = conv1x1(in_depth, config.img_depth)

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)

    def grow(self):
        raise NotImplementedError

    def forward(self, z_embs: Tensor):
        out = z_embs.view(*z_embs.shape, 1, 1)
        out = self.input_norm(out)
        out = self.stages(out)
        out = self.to_rgb(out)
        return out


def init_weights(module: nn.Module):
    if isinstance(module, (nn.modules.conv._ConvNd, nn.Linear)):
        nn.init.normal_(module.weight)
        parametrize.register_parametrization(module, "weight", EqualizedLR(module.weight))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
