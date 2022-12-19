# StyleGAN - https://arxiv.org/abs/1812.04948
# Not implemented features
# - Progressive growing and Equalized learning rate (from Progressive GAN)
#
# Code reference:
# https://github.com/NVlabs/stylegan

import math
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import Blur, _Act
from .progressive_gan import Discriminator


class GeneratorBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        w_dim: int,
        kernel_size: int,
        act: Optional[_Act] = partial(nn.LeakyReLU, 0.2, True),
        upsample: bool = False,
        demodulation: bool = True,
    ):
        super().__init__()
        self.demodulation = demodulation
        self.upsample = nn.Upsample(scale_factor=2.0) if upsample else None
        self.blur = Blur([1, 3, 3, 1]) if upsample else None

        self.weight = nn.Parameter(torch.empty(out_dim, in_dim, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(1, out_dim, 1, 1))
        self.noise_weight = nn.Parameter(torch.empty(1, out_dim, 1, 1))  # B in paper
        self.style_map = nn.Linear(w_dim, in_dim)  # A in paper
        self.act = act() if act is not None else nn.Identity()

        nn.init.kaiming_normal_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.noise_weight)
        nn.init.kaiming_normal_(self.style_map.weight)
        nn.init.ones_(self.style_map.bias)

    def forward(self, imgs: Tensor, w_embs: Tensor, noise: Optional[Tensor] = None):
        if self.upsample is not None:
            imgs = self.upsample(imgs)
        b, c, h, w = imgs.shape
        out_dim, in_dim, ky, kx = self.weight.shape

        # modulation
        weight = self.weight[None] * self.style_map(w_embs).view(b, 1, c, 1, 1)
        weight = weight.view(b * out_dim, in_dim, ky, kx)
        if self.demodulation:
            weight = F.normalize(weight, dim=(1, 2, 3))

        imgs = imgs.reshape(1, b * c, h, w)
        imgs = F.conv2d(imgs, weight, padding="same", groups=b)
        imgs = imgs.view(b, out_dim, h, w)
        if self.blur is not None:
            imgs = self.blur(imgs)

        noise = noise or torch.randn(b, 1, h, w, device=imgs.device)
        return self.act(imgs + self.bias + noise * self.noise_weight)

    def extra_repr(self) -> str:
        in_dim, out_dim, *kernel_size = self.weight.shape
        return f"in_dim={in_dim}, out_dim={out_dim}, kernel_size={kernel_size}"


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
        self.upsample_blur = nn.Sequential(nn.Upsample(scale_factor=2.0), Blur([1, 3, 3, 1]))

        block = partial(GeneratorBlock, w_dim=w_dim, kernel_size=3, act=act)
        to_rgb = partial(GeneratorBlock, out_dim=img_depth, w_dim=w_dim, kernel_size=1, demodulation=False, act=None)
        in_depth = learned_input_depth
        depth = base_depth * img_size // smallest_map_size
        out_depth = min(depth, max_depth)

        self.layers = nn.ModuleList()
        self.layers.append(block(in_depth, out_depth))
        self.layers.append(to_rgb(out_depth))
        in_depth = out_depth
        depth //= 2

        while smallest_map_size < img_size:
            out_depth = min(depth, max_depth)
            self.layers.append(block(in_depth, out_depth, upsample=True))
            self.layers.append(block(out_depth, out_depth))
            self.layers.append(to_rgb(out_depth))
            in_depth = out_depth
            depth //= 2
            smallest_map_size *= 2

        nn.init.ones_(self.learned_input)

    def forward(self, z_embs: Tensor, ys: Optional[Tensor] = None):
        w_embs = self.mapping_network(z_embs)
        x = self.learned_input.expand(z_embs.size(0), -1, -1, -1)
        y = 0
        for layer in self.layers:
            if not layer.demodulation:  # to_rgb layer
                if isinstance(y, Tensor):
                    y = self.upsample_blur(y)
                y = y + layer(x, w_embs)
            else:
                x = layer(x, w_embs)
        return y
