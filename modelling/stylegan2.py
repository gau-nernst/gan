# StyleGAN - https://arxiv.org/abs/1812.04948
# Not implemented features
# - R1 regularization, path length regularization
#
# Code reference:
# https://github.com/NVlabs/stylegan

import math
from dataclasses import dataclass, replace
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .nvidia_ops import Blur
from .progressive_gan import Discriminator, init_weights
from .stylegan import MappingNetwork, StyleGANConfig

Discriminator = partial(Discriminator, residual=True)


@dataclass
class StyleGAN2Config(StyleGANConfig):
    blur_size: int = 4  # override stylegan


class ModulatedConv2d(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int, w_dim: int, demodulation: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size)  # to contain weights and register parametrization
        self.style_weight = nn.Linear(w_dim, in_dim)  # A in paper
        self.demodulation = demodulation

    def forward(self, imgs: Tensor, w_embs: Tensor):
        b, c, h, w = imgs.shape
        out_dim, in_dim, ky, kx = self.conv.weight.shape

        # modulation
        style = self.style_weight(w_embs).view(b, 1, c, 1, 1) + 1
        weight = self.conv.weight.unsqueeze(0) * style
        weight = weight.view(b * out_dim, in_dim, ky, kx)
        if self.demodulation:
            weight = weight / torch.linalg.vector_norm(weight, dim=(1, 2, 3), keepdim=True)

        # use vmap?
        imgs = imgs.reshape(b * c, h, w)
        imgs = F.conv2d(imgs, weight, padding="same", groups=b)
        imgs = imgs.reshape(b, out_dim, h, w)

        if self.conv.bias is not None:
            imgs = imgs + self.conv.bias.view(1, -1, 1, 1)
        return imgs


class GeneratorBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, config: StyleGAN2Config, upsample: bool = False):
        super().__init__()
        # TODO: merge upsample with conv into conv_transpose (up_conv_blur)
        self.up = nn.Upsample(scale_factor=2.0) if upsample else nn.Identity()
        self.conv = ModulatedConv2d(in_dim, out_dim, 3, config.w_dim)
        self.blur = Blur(config.blur_size) if upsample else nn.Identity()
        self.noise_weight = nn.Parameter(torch.tensor(0.0))  # B in paper
        self.act = config.act()

    def forward(self, imgs: Tensor, w_embs: Tensor):
        imgs = self.blur(self.conv(self.up(imgs), w_embs))
        b, _, h, w = imgs.shape
        noise = torch.randn(b, 1, h, w, device=imgs.device)
        return self.act(imgs + noise * self.noise_weight)


class Generator(nn.Module):
    def __init__(self, config: Optional[StyleGAN2Config] = None, **kwargs):
        config = config or StyleGAN2Config()
        config = replace(config, **kwargs)
        assert config.img_size > 4 and math.log2(config.img_size).is_integer()
        super().__init__()
        self.mapping_network = MappingNetwork(config)

        map_size = config.smallest_map_size
        in_depth = config.input_depth
        self.learned_input = nn.Parameter(torch.empty(1, in_depth, map_size, map_size))
        self.up_blur = Blur(config.blur_size, up=2)

        depth = config.base_depth * config.img_size // map_size
        out_depth = min(depth, config.max_depth)

        self.first_block = GeneratorBlock(in_depth, out_depth, config)
        self.first_to_rgb = ModulatedConv2d(out_depth, config.img_depth, 1, config.w_dim, demodulation=False)
        in_depth = out_depth
        depth //= 2

        self.stages = nn.ModuleList()
        while map_size < config.img_size:
            out_depth = min(depth, config.max_depth)
            stage = dict(
                block1=GeneratorBlock(in_depth, out_depth, config, upsample=True),
                block2=GeneratorBlock(out_depth, out_depth, config),
                to_rgb=ModulatedConv2d(out_depth, config.img_depth, 1, config.w_dim, demodulation=False),
            )
            self.stages.append(nn.ModuleDict(stage))
            in_depth = out_depth
            depth //= 2
            map_size *= 2

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.learned_input)
        self.stages.apply(init_weights)

    def forward(self, z_embs: Tensor):
        w_embs = self.mapping_network(z_embs)

        x = self.learned_input.expand(z_embs.size(0), -1, -1, -1)
        x = self.first_block(x, w_embs[:, 0])
        y = self.first_to_rgb(x, w_embs[:, 1])

        for i, stage in enumerate(self.stages):
            x = stage["block1"](x, w_embs[:, (i + 1) * 2])
            x = stage["block2"](x, w_embs[:, (i + 1) * 2 + 1])
            y = stage["to_rgb"](x, w_embs[:, (i + 1) * 2 + 1]) + self.up_blur(y)

        return y
