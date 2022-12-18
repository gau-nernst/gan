# StyleGAN - https://arxiv.org/abs/1812.04948
# Not implemented features
# - BlurPool (https://arxiv.org/abs/1904.11486): helps to reduce anti-aliasing
# - Progressive growing and Equalized learning rate
#
# Code reference:
# https://github.com/NVlabs/stylegan

import math
from functools import partial
from typing import Callable, Optional

import torch
from torch import Tensor, nn

from .base import _Act
from .progressive_gan import Discriminator


class GeneratorBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        w_dim: int,
        conv: Optional[Callable[..., nn.Module]],
        act: _Act,
    ):
        super().__init__()
        self.conv = conv(in_dim, out_dim)
        self.noise_weight = nn.Parameter(torch.empty(1, out_dim, 1, 1))  # B in paper
        self.act = act()
        self.norm = nn.InstanceNorm2d(out_dim)
        self.style_map = nn.Linear(w_dim, out_dim * 2)  # A in paper

        if isinstance(self.conv, nn.modules.conv._ConvNd):
            nn.init.kaiming_normal_(self.conv.weight)
            nn.init.zeros_(self.conv.bias)
        nn.init.zeros_(self.noise_weight)
        nn.init.kaiming_normal_(self.style_map.weight)
        nn.init.ones_(self.style_map.bias[:out_dim])  # weight
        nn.init.zeros_(self.style_map.bias[out_dim:])  # bias

    def forward(self, imgs: Tensor, w_embs: Tensor):
        imgs = self.conv(imgs)
        b, c, h, w = imgs.shape

        noise = torch.randn(b, 1, h, w, device=imgs.device)
        imgs = self.act(imgs + noise * self.noise_weight)

        style = self.style_map(w_embs).view(b, -1, 1, 1)
        return self.norm(imgs) * style[:, :c] + style[:, c:]


class Generator(nn.Module):
    def __init__(
        self,
        img_size: int,
        img_depth: int = 3,
        z_dim: int = 512,
        w_dim: int = 512,
        mapping_network_depth: int = 8,
        learned_input_depth: int = 512,
        smallest_map_size: int = 4,
        base_depth: int = 16,
        max_depth: int = 512,
        act: _Act = partial(nn.LeakyReLU, 0.2, True),
    ):
        assert img_size > 4 and math.log2(img_size).is_integer()
        super().__init__()
        self.mapping_network = nn.Sequential()
        for i in range(mapping_network_depth):
            self.mapping_network.append(nn.Linear(z_dim if i == 0 else w_dim, w_dim))
            self.mapping_network.append(act())

        self.learned_input = nn.Parameter(torch.empty(1, learned_input_depth, smallest_map_size, smallest_map_size))

        conv3x3 = partial(nn.Conv2d, kernel_size=3, padding=1)
        up_conv = partial(nn.ConvTranspose2d, kernel_size=4, stride=2, padding=1)
        block = partial(GeneratorBlock, w_dim=w_dim, act=act)
        in_depth = learned_input_depth
        depth = base_depth * img_size // smallest_map_size
        out_depth = min(depth, max_depth)

        self.layers = nn.ModuleList()
        self.layers.append(block(in_depth, in_depth, conv=nn.Identity))
        self.layers.append(block(in_depth, out_depth, conv=conv3x3))
        in_depth = out_depth
        depth //= 2

        while smallest_map_size < img_size:
            out_depth = min(depth, max_depth)
            self.layers.append(block(in_depth, out_depth, conv=up_conv))
            self.layers.append(block(out_depth, out_depth, conv=conv3x3))
            in_depth = out_depth
            depth //= 2
            smallest_map_size *= 2

        self.out_conv = nn.Conv2d(in_depth, img_depth, 1)

        nn.init.ones_(self.learned_input)

    def forward(self, z_embs: Tensor, ys: Optional[Tensor] = None):
        w_embs = self.mapping_network(z_embs)
        x = self.learned_input.repeat(z_embs.size(0), 1, 1, 1)
        for layer in self.layers:
            x = layer(x, w_embs)
        return self.out_conv(x)
