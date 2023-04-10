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
from functools import partial
from typing import Optional

from torch import Tensor, nn
from torch.nn.utils import parametrize

from .base import _Act, _Norm, conv1x1, conv3x3, conv_norm_act
from .nvidia_ops import EqualizedLR, MinibatchStdDev, PixelNorm, blur_conv_down, up_conv_blur


class DiscriminatorStage(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        residual: bool = False,
        act: _Act = partial(nn.LeakyReLU, 0.2, True),
        blur_size: Optional[int] = None,
    ):
        super().__init__()
        conv_act = partial(conv_norm_act, order=["conv", "act"], conv=conv3x3, act=act)
        down_conv = partial(blur_conv_down, blur_size=blur_size)

        self.main = nn.Sequential(
            conv_act(in_dim, in_dim),
            conv_act(in_dim, out_dim, conv=partial(down_conv, kernel_size=3)),
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
        blur_size: Optional[int] = None,
    ):
        assert img_size > 4 and math.log2(img_size).is_integer()
        super().__init__()
        conv_act = partial(conv_norm_act, order=["conv", "act"], conv=conv3x3, act=act)
        stage = partial(DiscriminatorStage, act=act, residual=residual, blur_size=blur_size)

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

    def grow(self):
        raise NotImplementedError

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
        blur_size: Optional[int] = None,
    ):
        assert img_size > 4 and math.log2(img_size).is_integer()
        super().__init__()
        up_conv3x3 = partial(up_conv_blur, kernel_size=3, blur_size=blur_size)
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

    def grow(self):
        raise NotImplementedError

    def forward(self, z_embs: Tensor):
        return self.layers(z_embs.view(*z_embs.shape, 1, 1))


def init_weights(module: nn.Module):
    if isinstance(module, (nn.modules.conv._ConvNd, nn.Linear)):
        nn.init.normal_(module.weight)
        parametrize.register_parametrization(module, "weight", EqualizedLR(module.weight))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
