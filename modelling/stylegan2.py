# StyleGAN - https://arxiv.org/abs/1812.04948
# Not implemented features
# - BlurPool (https://arxiv.org/abs/1904.11486): helps to reduce anti-aliasing
# - Progressive growing and Equalized learning rate
#
# Code reference:
# https://github.com/NVlabs/stylegan

import math
from functools import partial
from typing import Callable, Optional, Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .base import _Act
from .progressive_gan import Discriminator


def upsample(imgs: Tensor):
    return F.interpolate(imgs, scale_factor=2.0, mode="bilinear", antialias=True)


def downsample(imgs: Tensor):
    return F.interpolate(imgs, scale_factor=0.5, mode="bilinear", antialias=True)


def get_resize(resize: Literal["none", "up", "down"]):
    if resize == "none":
        return nn.Identity()
    return dict(up=upsample, down=downsample)[resize]


class GeneratorBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        w_dim: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        resize: Literal["none", "up", "down"] = "none",
        demodulation: bool = True,
        act: Optional[_Act] = partial(nn.LeakyReLU, 0.2, True),
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.resize = get_resize(resize)
        self.demodulation = demodulation

        self.weight = nn.Parameter(torch.empty(out_dim, in_dim, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_dim))
        self.noise_weight = nn.Parameter(torch.empty(1, out_dim, 1, 1)) # B in paper
        self.style_map = nn.Linear(w_dim, in_dim)    # A in paper
        self.act = act() if act is not None else nn.Identity()

        nn.init.kaiming_normal_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.noise_weight)
        nn.init.kaiming_normal_(self.style_map.weight)
        nn.init.ones_(self.style_map.bias)

    def forward(self, imgs: Tensor, w_embs: Tensor):
        imgs = self.resize(imgs)
        b, c, h, w = imgs.shape

        # modulation
        weight = self.weight[None] * self.style_map(w_embs).view(b, 1, c, 1, 1)
        weight = weight.view(b * self.out_dim, self.in_dim, self.kernel_size, self.kernel_size)
        if self.demodulation:
            weight = F.normalize(weight, dim=(1, 2, 3))

        imgs = imgs.reshape(1, b * c, h, w)
        imgs = F.conv2d(imgs, weight, self.bias.repeat(b), self.stride, self.padding, groups=b)
        imgs = imgs.view(b, self.out_dim, h, w)

        noise = torch.randn(b, 1, h, w, device=imgs.device)
        return self.act(imgs + noise * self.noise_weight)
    
    def extra_repr(self) -> str:
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}, kernel_size={self.kernel_size}"


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

        block = partial(GeneratorBlock, w_dim=w_dim, kernel_size=3, padding=1, act=act)
        up_block = partial(GeneratorBlock, w_dim=w_dim, kernel_size=3, padding=1, resize="up", act=act)
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
            print(in_depth)
            out_depth = min(depth, max_depth)
            self.layers.append(up_block(in_depth, out_depth))
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
                y = upsample(y) if isinstance(y, Tensor) else y
                y = y + layer(x, w_embs)
            else:
                x = layer(x, w_embs)
        return y
