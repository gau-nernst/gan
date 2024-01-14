# StyleGAN - https://arxiv.org/abs/1812.04948
# Not implemented features
# - Progressive growing (from Progressive GAN)
# - Style mixing, truncation in W space
#
# Code reference:
# https://github.com/NVlabs/stylegan

import math

import torch
from torch import Tensor, nn

from .progressive_gan import init_weights


class ApplyNoise(nn.Module):
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


class StyleGanGeneratorBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, z_dim: int, residual: bool = False) -> None:
        super().__init__()
        residual = [
            AdaIN(in_dim, z_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2.0),
            nn.Conv2d(in_dim, out_dim, 3, 1, 1),
            # TODO: blur layer
            ApplyNoise(out_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_dim, out_dim, 3, 1, 1),
            ApplyNoise(out_dim),
        ]
        self.residual = nn.ModuleList(residual)
        if residual:
            self.shortcut = nn.Sequential(nn.Conv2d(in_dim, out_dim, 1), nn.Upsample(scale_factor=2.0))
            self.scale = nn.Parameter(torch.full((out_dim, 1, 1), 1e-4))
        else:
            self.shortcut = None

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        out = x
        for layer in self.residual:
            out = layer(out, w) if isinstance(layer, AdaIN) else layer(out)
        if self.shortcut is not None:
            out = self.shortcut(x) + out * self.scale
        return out


class StyleGanGenerator(nn.Module):
    def __init__(
        self,
        img_size: int,
        img_channels: int = 3,
        z_dim: int = 128,
        base_dim: int = 16,
        mapping_network_depth: int = 8,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.mapping_network = nn.Sequential(nn.LayerNorm(z_dim))
        for _ in range(mapping_network_depth):
            self.mapping_network.extend([nn.Linear(z_dim, z_dim), nn.LeakyReLU(0.2, inplace=True)])

        self.learned_input = nn.Parameter(torch.ones(1, 512, 4, 4))
        self.blocks = nn.ModuleList()
        in_ch = 512

        depth = int(math.log2(img_size // 4))
        for i in range(depth):
            out_ch = min(base_dim * img_size // 4 // 2 ** (i + 1), 512)
            self.blocks.append(StyleGanGeneratorBlock(in_ch, out_ch, z_dim, residual=residual))
            in_ch = out_ch

        self.out_conv = nn.Sequential(
            nn.InstanceNorm2d(in_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_ch, img_channels, 1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)

    def forward(self, z: Tensor):
        w = self.mapping_network(z)
        x = self.learned_input.expand(z.shape[0], -1, -1, -1)
        for block in self.blocks:
            x = block(x, w)
        return self.out_conv(x)
