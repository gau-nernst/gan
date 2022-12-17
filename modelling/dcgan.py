# DCGAN - https://arxiv.org/abs/1701.07875
# Discriminator will downsample until the feature map is 4x4, then flatten and matmul
# Generator will matmul, reshape to 4x4, then upsample until the desired image size is obtained
#
# Code references:
# https://github.com/soumith/dcgan.torch/
# https://github.com/martinarjovsky/WassersteinGAN/

import math
from functools import partial
from typing import Optional

import torch
from torch import Tensor, nn

from .base import _Act, _Norm, conv_norm_act


class Discriminator(nn.Module):
    def __init__(
        self,
        img_size: int = 64,
        img_depth: int = 3,
        smallest_map_size: int = 4,
        base_depth: int = 64,
        norm: _Norm = nn.BatchNorm2d,
        act: _Act = partial(nn.LeakyReLU, 0.2, True),
        c_encoder: Optional[nn.Module] = None,
        c_dim: int = 0,
    ):
        assert img_size >= 4 and math.log2(img_size).is_integer()
        assert math.log2(smallest_map_size).is_integer()
        if c_encoder is not None:
            assert c_dim > 0
            img_depth += c_dim
        super().__init__()
        layer = partial(conv_norm_act, kernel_size=4, stride=2, padding=1, norm=norm, act=act)

        self.c_encoder = c_encoder
        self.layers = nn.Sequential()
        self.layers.append(layer(img_depth, base_depth))
        img_size /= 2

        # add strided conv until image size = 4
        while img_size > smallest_map_size:
            self.layers.append(layer(base_depth, base_depth * 2))
            base_depth *= 2
            img_size //= 2

        # flatten and matmul
        self.layers.append(nn.Conv2d(base_depth, 1, smallest_map_size))

        self.layers.apply(init_weights)

    def forward(self, imgs: Tensor, ys: Optional[Tensor] = None):
        if self.c_encoder is not None:
            assert ys is not None
            img_h, img_w = imgs.shape[2:]
            y_embs = self.c_encoder(ys)[:, :, None, None].expand(-1, -1, img_h, img_w)
            imgs = torch.cat([imgs, y_embs], dim=1)
        return self.layers(imgs)


class Generator(nn.Module):
    def __init__(
        self,
        img_size: int = 64,
        img_depth: int = 3,
        z_dim: int = 128,
        smallest_map_size: int = 4,
        base_depth: int = 64,
        norm: _Norm = nn.BatchNorm2d,
        act: _Act = partial(nn.ReLU, True),
        c_encoder: Optional[nn.Module] = None,
        c_dim: int = 0,
    ):
        assert img_size >= 4 and math.log2(img_size).is_integer()
        assert math.log2(smallest_map_size).is_integer()
        if c_encoder is not None:
            assert c_dim >= 0
            z_dim += c_dim
        super().__init__()
        kwargs = dict(conv=nn.ConvTranspose2d, norm=norm, act=act)

        self.c_encoder = c_encoder
        self.layers = nn.Sequential()

        # matmul and reshape to 4x4
        depth = base_depth * img_size // 2 // smallest_map_size
        self.layers.append(conv_norm_act(z_dim, depth, smallest_map_size, **kwargs))

        # conv transpose until reaching image size / 2
        kwargs.update(kernel_size=4, stride=2, padding=1)
        while smallest_map_size < img_size // 2:
            self.layers.append(conv_norm_act(depth, depth // 2, **kwargs))
            depth //= 2
            smallest_map_size *= 2

        # last layer use tanh activation
        kwargs.update(norm=None, act=nn.Tanh)
        self.layers.append(conv_norm_act(depth, img_depth, **kwargs))

        self.layers.apply(init_weights)

    def forward(self, z_embs: Tensor, ys: Optional[Tensor] = None):
        if self.c_encoder is not None:
            assert ys is not None
            y_embs = self.c_encoder(ys)
            z_embs = torch.cat([z_embs, y_embs], dim=1)
        return self.layers(z_embs[:, :, None, None])


def init_weights(module: nn.Module):
    if isinstance(module, nn.modules.conv._ConvNd):
        nn.init.normal_(module.weight, 0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.modules.batchnorm._BatchNorm):
        nn.init.normal_(module.weight, 1, 0.02)
        nn.init.zeros_(module.bias)
