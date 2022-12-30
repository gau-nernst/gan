# StyleGAN - https://arxiv.org/abs/1812.04948
# Not implemented features
# - Progressive growing and Equalized learning rate (from Progressive GAN)
#
# Code reference:
# https://github.com/NVlabs/stylegan

import math
from functools import partial
from typing import List, Optional

import torch
from torch import Tensor, nn

from .base import _Act, _Norm, conv1x1, conv3x3
from .progressive_gan import Blur, Discriminator, init_weights


class GeneratorBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        w_dim: int,
        first_block: bool = False,
        upsample: bool = False,
        norm: _Norm = nn.InstanceNorm2d,
        act: _Act = partial(nn.LeakyReLU, 0.2, True),
        blur_kernel: Optional[List[float]] = None,
    ):
        blur_kernel = blur_kernel or [1, 2, 1]
        super().__init__()
        if first_block:
            self.conv = nn.Identity()
        elif upsample:
            self.conv = nn.Sequential(
                nn.Upsample(scale_factor=2.0),
                conv3x3(in_dim, out_dim),
                Blur(blur_kernel),
            )
        else:
            self.conv = conv3x3(in_dim, out_dim)
        self.noise_weight = nn.Parameter(torch.zeros(1, out_dim, 1, 1))  # B in paper
        self.act = act()
        self.norm = norm(out_dim)
        self.style_map = nn.Linear(w_dim, out_dim * 2)  # A in paper

    def forward(self, imgs: Tensor, w_embs: Tensor, noise: Optional[Tensor] = None):
        imgs = self.conv(imgs)
        b, c, h, w = imgs.shape

        noise = noise or torch.randn(b, 1, h, w, device=imgs.device)
        imgs = self.act(imgs + noise * self.noise_weight)

        style = self.style_map(w_embs).view(b, -1, 1, 1)
        return self.norm(imgs) * style[:, :c].add(1) + style[:, c:]


class Generator(nn.Module):
    def __init__(
        self,
        img_size: int,
        img_depth: int = 3,
        z_dim: int = 512,
        w_dim: int = 512,
        mapping_network_depth: int = 8,
        input_depth: int = 512,
        smallest_map_size: int = 4,
        base_depth: int = 16,
        max_depth: int = 512,
        act: _Act = partial(nn.LeakyReLU, 0.2, True),
        blur_kernel: Optional[List[float]] = None,
    ):
        assert img_size > 4 and math.log2(img_size).is_integer()
        super().__init__()
        self.mapping_network = nn.Sequential()
        for i in range(mapping_network_depth):
            self.mapping_network.append(nn.Linear(z_dim if i == 0 else w_dim, w_dim))
            self.mapping_network.append(act())

        self.learned_input = nn.Parameter(torch.empty(1, input_depth, smallest_map_size, smallest_map_size))

        block = partial(GeneratorBlock, w_dim=w_dim, act=act, blur_kernel=blur_kernel)
        depth = base_depth * img_size // smallest_map_size
        out_depth = min(depth, max_depth)

        self.layers = nn.ModuleList()
        self.layers.append(block(input_depth, input_depth, first_block=True))
        self.layers.append(block(input_depth, out_depth))
        input_depth = out_depth
        depth //= 2

        while smallest_map_size < img_size:
            out_depth = min(depth, max_depth)
            self.layers.append(block(input_depth, out_depth, upsample=True))
            self.layers.append(block(out_depth, out_depth))
            input_depth = out_depth
            depth //= 2
            smallest_map_size *= 2

        self.out_conv = conv1x1(input_depth, img_depth)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.learned_input)
        self.layers.apply(init_weights)

    def forward(self, z_embs: Tensor):
        w_embs = self.mapping_network(z_embs)
        x = self.learned_input.repeat(z_embs.size(0), 1, 1, 1)
        for layer in self.layers:
            x = layer(x, w_embs)
        return self.out_conv(x)
