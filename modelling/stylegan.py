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
        super().__init__(dim, eps=1e-8)
        self.style_w = nn.Linear(z_dim, dim)
        self.style_b = nn.Linear(z_dim, dim)

        with torch.no_grad():
            self.style_w.bias.add_(1)

    def forward(self, x: Tensor, w_embs: Tensor) -> Tensor:
        weight = self.style_w(w_embs).unflatten(-1, (-1, 1, 1))
        bias = self.style_b(w_embs).unflatten(-1, (-1, 1, 1))
        return super().forward(x) * weight + bias


class StyleGanGeneratorBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, z_dim: int, first_block: bool = False) -> None:
        super().__init__()
        self.residual = nn.ModuleList()
        self.shortcut = nn.Sequential(nn.Conv2d(in_dim, out_dim, 1))
        if not first_block:
            self.residual.append(nn.Upsample(scale_factor=2.0))
            self.residual.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
            # TODO: blur layer
            in_dim = out_dim

            self.shortcut.append(nn.Upsample(scale_factor=2.0))

        self.residual.extend(
            [
                ApplyNoise(in_dim),
                nn.LeakyReLU(0.2, inplace=True),
                AdaIN(in_dim, z_dim),
                nn.Conv2d(in_dim, out_dim, 3, 1, 1),
                ApplyNoise(out_dim),
                nn.LeakyReLU(0.2, inplace=True),
                AdaIN(out_dim, z_dim),
            ]
        )

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        for layer in self.residual:
            x = layer(x, w) if isinstance(layer, AdaIN) else layer(x)
        return x


class StyleGanGenerator(nn.Module):
    def __init__(self, img_size: int, img_channels: int = 3, z_dim: int = 128, base_dim: int = 16) -> None:
        super().__init__()
        self.mapping_network = nn.Sequential(nn.LayerNorm(z_dim))
        for _ in range(8):
            self.mapping_network.extend([nn.Linear(z_dim, z_dim), nn.LeakyReLU(0.2, inplace=True)])

        self.learned_input = nn.Parameter(torch.ones(1, 512, 4, 4))

        out_ch = min(base_dim * img_size // 4, 512)
        self.blocks = nn.ModuleList()
        self.blocks.append(StyleGanGeneratorBlock(512, out_ch, z_dim, first_block=True))
        in_ch = out_ch

        depth = int(math.log2(img_size // 4))
        for i in range(depth):
            out_ch = min(base_dim * img_size // 4 // 2 ** (i + 1), 512)
            self.blocks.append(StyleGanGeneratorBlock(in_ch, out_ch, z_dim))
            in_ch = out_ch

        self.out_conv = nn.Conv2d(in_ch, img_channels, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)

    def forward(self, z: Tensor):
        w = self.mapping_network(z)
        x = self.learned_input.expand(z.shape[0], -1, -1, -1)
        for block in self.blocks:
            x = block(x, w)
        return self.out_conv(x)
