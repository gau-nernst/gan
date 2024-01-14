# StyleGAN2 - https://arxiv.org/abs/1912.04958
# Not implemented features
# - Path length regularization
#
# Code reference:
# https://github.com/NVlabs/stylegan2

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .progressive_gan import init_weights
from .stylegan import ApplyNoise


batched_conv2d = torch.vmap(F.conv2d)


class ModulatedConv2d(nn.Conv2d):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, z_dim: int, demodulation: bool = True
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)
        self.style_weight = nn.Linear(z_dim, in_channels)  # A in paper
        self.demodulation = demodulation

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        style = self.style_weight(w).unflatten(-1, (1, -1, 1, 1)) + 1
        weight = self.weight.unsqueeze(0) * style
        if self.demodulation:
            weight = F.normalize(weight, dim=(2, 3, 4), eps=1e-8)

        bias = self.bias.view(1, -1).expand(x.shape[0], -1)
        return batched_conv2d(x, weight, bias, padding=self.padding)


class StyleGan2GeneratorBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, z_dim: int, img_channels: int, first_block: bool = False) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        if not first_block:
            layers = [
                nn.Upsample(scale_factor=2.0),
                ModulatedConv2d(in_dim, out_dim, 3, z_dim),
                # TODO: blur
                ApplyNoise(out_dim),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            self.layers.extend(layers)

        layers = [
            ModulatedConv2d(out_dim, out_dim, 3, z_dim),
            ApplyNoise(out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        self.layers.extend(layers)
        self.to_rgb = ModulatedConv2d(out_dim, img_channels, 1, z_dim, demodulation=False)

    def forward(self, x: Tensor, w_embs: Tensor):
        for layer in self.layers:
            x = layer(x, w_embs) if isinstance(layer, ModulatedConv2d) else layer(x)
        return x, self.to_rgb(x)


class StyleGan2Generator(nn.Module):
    def __init__(self, img_size: int, img_channels: int = 3, z_dim: int = 128, base_dim: int = 16) -> None:
        super().__init__()
        self.mapping_network = nn.Sequential(nn.LayerNorm(z_dim))
        for _ in range(8):
            self.mapping_network.extend([nn.Linear(z_dim, z_dim), nn.LeakyReLU(0.2, inplace=True)])

        self.learned_input = nn.Parameter(torch.ones(1, 512, 4, 4))

        out_ch = min(base_dim * img_size // 4, 512)
        self.blocks = nn.ModuleList()
        self.blocks.append(StyleGan2GeneratorBlock(512, out_ch, z_dim, img_channels, first_block=True))
        in_ch = out_ch

        depth = int(math.log2(img_size // 4))
        for i in range(depth):
            out_ch = min(base_dim * img_size // 4 // 2 ** (i + 1), 512)
            self.blocks.append(StyleGan2GeneratorBlock(in_ch, out_ch, z_dim, img_channels))
            in_ch = out_ch

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)

    def forward(self, z: Tensor):
        w = self.mapping_network(z)
        x = self.learned_input.expand(z.shape[0], -1, -1, -1)
        x, y = self.blocks[0](x, w)

        for block in self.blocks[1:]:
            x, new_y = block(x, w)
            y = F.upsample(y, scale_factor=2.0) + new_y

        return y
