# StyleGAN2 - https://arxiv.org/abs/1912.04958
# Not implemented features
# - Path length regularization
#
# Code reference:
# https://github.com/NVlabs/stylegan2

import math
from dataclasses import dataclass, replace
from typing import Optional

import torch
from torch import Tensor, nn

from .base import batched_conv2d
from .nvidia_ops import Blur
from .progressive_gan import Discriminator, init_weights
from .stylegan import MappingNetwork, StyleGANConfig


@dataclass
class StyleGAN2Config(StyleGANConfig):
    blur_size: int = 4  # override stylegan


class Discriminator(Discriminator):
    def __init__(self, config: Optional[StyleGAN2Config] = None, **kwargs):
        config = config or StyleGAN2Config()
        config = replace(config, **kwargs)
        super().__init__(
            img_size=config.img_size,
            img_depth=config.img_depth,
            base_depth=config.base_depth,
            max_depth=config.max_depth,
            smallest_map_size=config.smallest_map_size,
            residual=True,
            act=config.act,
            blur_size=config.blur_size,
        )


class ModulatedConv2d(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int, w_dim: int, demodulation: bool = True):
        super().__init__()
        # to contain weights for initialization and parametrization
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, padding=(kernel_size - 1) // 2)
        self.style_weight = nn.Linear(w_dim, in_dim)  # A in paper
        self.demodulation = demodulation

    def forward(self, imgs: Tensor, w_embs: Tensor):
        b, c, _, _ = imgs.shape

        # modulation
        style = self.style_weight(w_embs).view(b, 1, c, 1, 1) + 1
        weight = self.conv.weight.unsqueeze(0) * style
        if self.demodulation:
            weight = weight / torch.linalg.vector_norm(weight, dim=(2, 3, 4), keepdim=True)

        bias = self.conv.bias.view(1, -1).expand(b, -1) if self.conv.bias is not None else None
        imgs = batched_conv2d(imgs, weight, bias, padding=self.conv.padding)
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


class RGBLayer(ModulatedConv2d):
    def __init__(self, in_dim: int, config: StyleGAN2Config):
        super().__init__(in_dim, config.img_depth, 1, config.w_dim, demodulation=False)


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
        self.first_to_rgb = RGBLayer(out_depth, config)
        in_depth = out_depth
        depth //= 2

        self.stages = nn.ModuleList()
        while map_size < config.img_size:
            out_depth = min(depth, config.max_depth)
            stage = dict(
                block1=GeneratorBlock(in_depth, out_depth, config, upsample=True),
                block2=GeneratorBlock(out_depth, out_depth, config),
                to_rgb=RGBLayer(out_depth, config),
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
