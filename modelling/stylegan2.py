# StyleGAN - https://arxiv.org/abs/1812.04948
# Not implemented features
# - Equalized learning rate (from Progressive GAN)
# - Style mixing regularization, R1 regularization, path length regularization
#
# Code reference:
# https://github.com/NVlabs/stylegan

import math
from functools import partial
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import _Act
from .progressive_gan import Blur
from .progressive_gan import Discriminator as _Discriminator
from .progressive_gan import init_weights

Discriminator = partial(_Discriminator, residual=True)


class GeneratorBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        w_dim: int,
        kernel_size: int,
        upsample: bool = False,
        demodulation: bool = True,
        act: Optional[_Act] = partial(nn.LeakyReLU, 0.2, True),
        blur_kernel: Optional[List[float]] = None,
    ):
        blur_kernel = blur_kernel or [1, 3, 3, 1]
        super().__init__()
        self.demodulation = demodulation
        self.upsample = nn.Upsample(scale_factor=2.0) if upsample else None
        self.blur = Blur(blur_kernel) if upsample else None

        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size)
        self.noise_weight = nn.Parameter(torch.tensor(0.0))  # B in paper
        self.style_weight = nn.Linear(w_dim, in_dim)  # A in paper
        self.act = act() if act is not None else nn.Identity()

    def forward(self, imgs: Tensor, w_embs: Tensor, noise: Optional[Tensor] = None):
        if self.upsample is not None:
            imgs = self.upsample(imgs)
        b, c, h, w = imgs.shape
        out_dim, in_dim, ky, kx = self.conv.weight.shape

        # modulation
        style = self.style_weight(w_embs).view(b, 1, c, 1, 1) + 1
        weight = self.conv.weight.unsqueeze(0) * style
        weight = weight.view(b * out_dim, in_dim, ky, kx)
        if self.demodulation:
            weight = weight / torch.linalg.vector_norm(weight, dim=(1, 2, 3), keepdim=True)

        imgs = imgs.reshape(1, b * c, h, w)
        imgs = F.conv2d(imgs, weight, padding="same", groups=b)
        imgs = imgs.view(b, out_dim, h, w)
        if self.blur is not None:
            imgs = self.blur(imgs)

        noise = noise or torch.randn(b, 1, h, w, device=imgs.device)
        return self.act(imgs + self.conv.bias.view(1, -1, 1, 1) + noise * self.noise_weight)


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
        blur_kernel = blur_kernel or [1, 3, 3, 1]
        super().__init__()
        self.mapping_network = nn.Sequential()
        for i in range(mapping_network_depth):
            self.mapping_network.append(nn.Linear(z_dim if i == 0 else w_dim, w_dim))
            self.mapping_network.append(act())

        self.learned_input = nn.Parameter(torch.empty(1, input_depth, smallest_map_size, smallest_map_size))
        self.upsample_blur = nn.Sequential(nn.Upsample(scale_factor=2.0), Blur(blur_kernel))

        block = partial(GeneratorBlock, w_dim=w_dim, kernel_size=3, act=act)
        to_rgb = partial(GeneratorBlock, out_dim=img_depth, w_dim=w_dim, kernel_size=1, demodulation=False, act=None)
        depth = base_depth * img_size // smallest_map_size
        out_depth = min(depth, max_depth)

        self.layers = nn.ModuleList()
        self.layers.append(block(input_depth, out_depth))
        self.layers.append(to_rgb(out_depth))
        input_depth = out_depth
        depth //= 2

        while smallest_map_size < img_size:
            out_depth = min(depth, max_depth)
            self.layers.append(block(input_depth, out_depth, upsample=True))
            self.layers.append(block(out_depth, out_depth))
            self.layers.append(to_rgb(out_depth))
            input_depth = out_depth
            depth //= 2
            smallest_map_size *= 2

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.learned_input)
        self.layers.apply(init_weights)

    def forward(self, z_embs: Tensor):
        w_embs = self.mapping_network(z_embs)
        x = self.learned_input.expand(z_embs.size(0), -1, -1, -1)
        y = 0
        for i, layer in enumerate(self.layers):
            if not layer.demodulation:  # to_rgb layer
                y = y + layer(x, w_embs)
                if i < len(self.layers) - 1:
                    y = self.upsample_blur(y)
            else:
                x = layer(x, w_embs)
        return y
