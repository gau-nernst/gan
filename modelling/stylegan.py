# StyleGAN - https://arxiv.org/abs/1812.04948
# Not implemented features
# - Progressive growing (from Progressive GAN)
# - Style mixing, truncation in W space
#
# Code reference:
# https://github.com/NVlabs/stylegan

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .progressive_gan import init_weights


class Noise(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(dim, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        N, _, H, W = x.shape
        noise = torch.randn(N, 1, H, W, device=x.device, dtype=x.dtype)
        return x + noise * self.scale


class AdaIN(nn.InstanceNorm2d):
    def __init__(self, dim: int, z_dim: int) -> None:
        super().__init__(dim)
        self.style_w = nn.Linear(z_dim, dim)
        self.style_b = nn.Linear(z_dim, dim)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        weight = self.style_w(w).unflatten(-1, (-1, 1, 1)) + 1
        bias = self.style_b(w).unflatten(-1, (-1, 1, 1))
        return super().forward(x) * weight + bias


class StyleGanGeneratorBlock(nn.ModuleList):
    def __init__(self, in_dim: int, out_dim: int, z_dim: int) -> None:
        layers = [
            AdaIN(in_dim, z_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2.0),
            nn.Conv2d(in_dim, out_dim, 3, 1, 1),
            Noise(out_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_dim, out_dim, 3, 1, 1),
            Noise(out_dim),
        ]
        super().__init__(layers)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        for layer in self:
            x = layer(x, w) if isinstance(layer, AdaIN) else layer(x)
        return x


class StyleGanGenerator(nn.Module):
    def __init__(
        self, img_size: int, img_channels: int = 3, z_dim: int = 128, base_dim: int = 16, mapping_network_depth: int = 8
    ) -> None:
        super().__init__()
        self.mapping_network = nn.Sequential(nn.LayerNorm(z_dim))
        for _ in range(mapping_network_depth):
            self.mapping_network.extend([nn.Linear(z_dim, z_dim), nn.LeakyReLU(0.2, inplace=True)])

        self.learned_input = nn.Parameter(torch.ones(1, 512, 4, 4))
        self.blocks = nn.ModuleList()
        self.out_convs = nn.ModuleList()
        in_ch = 512

        depth = int(math.log2(img_size // 4))
        for i in range(depth):
            out_ch = min(base_dim * img_size // 4 // 2 ** (i + 1), 512)
            self.blocks.append(StyleGanGeneratorBlock(in_ch, out_ch, z_dim))
            in_ch = out_ch
            self.out_convs.append(nn.Conv2d(in_ch, img_channels, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)

    def forward(self, z: Tensor):
        w = self.mapping_network(z) * 0.01
        x = self.learned_input.expand(z.shape[0], -1, -1, -1)

        x = self.blocks[0](x, w)
        out = self.out_convs[0](x)
        for block, out_conv in zip(self.blocks[1:], self.out_convs[1:]):
            x = block(x, w)
            out = F.interpolate(out, scale_factor=2.0) + out_conv(x)
        return out
