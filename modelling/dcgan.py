# DCGAN - https://arxiv.org/abs/1701.07875
# Discriminator will downsample until the feature map is 4x4, then flatten and matmul
# Generator will matmul, reshape to 4x4, then upsample until the desired image size is obtained
#
# Code references:
# https://github.com/soumith/dcgan.torch/
# https://github.com/martinarjovsky/WassersteinGAN/

import math
from functools import partial

from torch import Tensor, nn

from .base import _Act, _Norm, conv_norm_act


class Discriminator(nn.Module):
    def __init__(
        self,
        img_size: int = 64,
        img_channels: int = 3,
        init_map_size: int = 4,
        min_channels: int = 64,
        norm: _Norm = partial(nn.BatchNorm2d, track_running_stats=False),
        act: _Act = partial(nn.LeakyReLU, 0.2, True),
    ):
        assert img_size >= 4 and math.log2(img_size).is_integer()
        assert math.log2(init_map_size).is_integer()
        super().__init__()
        conv = partial(nn.Conv2d, kernel_size=4, stride=2, padding=1, bias=False)
        _conv_norm_act = partial(conv_norm_act, conv=conv, norm=norm, act=act)

        self.layers = nn.Sequential()
        self.layers.append(_conv_norm_act(img_channels, min_channels))
        img_size //= 2

        # add strided conv until image size = 4
        while img_size > init_map_size:
            self.layers.append(_conv_norm_act(min_channels, min_channels * 2))
            min_channels *= 2
            img_size //= 2

        # flatten and matmul
        self.layers.append(nn.Conv2d(min_channels, 1, init_map_size))

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)

    def forward(self, imgs: Tensor):
        return self.layers(imgs).view(-1)


class Generator(nn.Module):
    def __init__(
        self,
        img_size: int = 64,
        img_channels: int = 3,
        z_dim: int = 128,
        init_map_size: int = 4,
        min_channels: int = 64,
        norm: _Norm = partial(nn.BatchNorm2d, track_running_stats=False),
        act: _Act = partial(nn.ReLU, True),
    ):
        assert img_size >= 4 and math.log2(img_size).is_integer()
        assert math.log2(init_map_size).is_integer()
        super().__init__()
        conv = partial(nn.ConvTranspose2d, kernel_size=4, stride=2, padding=1, bias=False)
        _conv_norm_act = partial(conv_norm_act, conv=conv, norm=norm, act=act)
        channels = min_channels * img_size // 2 // init_map_size

        self.layers = nn.Sequential()

        # matmul and reshape to 4x4
        first_layer = _conv_norm_act(z_dim, channels, conv=partial(nn.ConvTranspose2d, kernel_size=init_map_size))
        self.layers.append(first_layer)

        # conv transpose until reaching image size / 2
        while init_map_size < img_size // 2:
            self.layers.append(_conv_norm_act(channels, channels // 2))
            channels //= 2
            init_map_size *= 2

        # last layer use tanh activation
        self.layers.append(
            nn.ConvTranspose2d(channels, img_channels, 4, 2, 1)
        )  # TODO: check if the last layer has bias
        self.layers.append(nn.Tanh())

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)

    def forward(self, z_embs: Tensor):
        return self.layers(z_embs[:, :, None, None])


def init_weights(module: nn.Module):
    if isinstance(module, nn.modules.conv._ConvNd):
        nn.init.normal_(module.weight, 0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.modules.batchnorm._BatchNorm):
        nn.init.normal_(module.weight, 1, 0.02)
        nn.init.zeros_(module.bias)
