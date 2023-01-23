# StyleGAN - https://arxiv.org/abs/1812.04948
# Not implemented features
# - Progressive growing (from Progressive GAN)
# - R1 regularization
#
# Code reference:
# https://github.com/NVlabs/stylegan

import math
from dataclasses import dataclass, replace
from functools import partial
from typing import Optional

import torch
from torch import Tensor, nn

from .base import _Act, conv1x1, conv3x3
from .nvidia_ops import up_conv_blur
from .progressive_gan import Discriminator, init_weights


@dataclass
class Config:
    img_size: int = 128  # basics
    img_depth: int = 3
    z_dim: int = 512
    w_dim: int = 512
    mapping_network_depth: int = 8  # mapping network
    w_mean_beta: float = 0.995
    style_mixing: float = 0.9
    truncation_psi: float = 0.7
    truncation_cutoff: int = 8
    smallest_map_size: int = 4  # synthesis network
    input_depth: int = 512
    base_depth: int = 16
    max_depth: int = 512
    act: _Act = partial(nn.LeakyReLU, 0.2, True)  # others
    blur_size: int = 3


class MappingNetwork(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.mlp = nn.Sequential()
        for i in range(config.mapping_network_depth):
            self.mlp.append(nn.Linear(config.z_dim if i == 0 else config.w_dim, config.w_dim))
            self.mlp.append(config.act())
        self.register_buffer("w_mean", torch.zeros(config.w_dim))
        self.w_mean: Tensor
        self.w_mean_beta = config.w_mean_beta
        self.style_mixing = config.style_mixing
        self.truncation_psi = config.truncation_psi
        self.truncation_cutoff = config.truncation_cutoff
        self.n_layers = int(math.log2(config.img_size // config.smallest_map_size) + 1) * 2

    def forward(self, z_embs: Tensor):
        w_embs = self.mlp(z_embs)

        if self.training:
            self.w_mean.lerp_(w_embs.detach().mean(0), 1 - self.w_mean_beta)
            w_embs = w_embs.unsqueeze(1).expand(-1, self.n_layers, -1)

            if self.style_mixing > 0 and torch.rand([]) < self.style_mixing:
                w_embs2 = self.mlp(torch.randn_like(z_embs))
                w_embs2 = w_embs2.unsqueeze(1).expand(-1, self.n_layers, -1)

                cutfoff = torch.arange(self.n_layers) < torch.randint(1, self.n_layers, (self.n_layers,))
                cutfoff = cutfoff.view(1, self.n_layers, 1).to(w_embs.device)
                w_embs = torch.where(cutfoff, w_embs, w_embs2)

        else:
            w_embs = w_embs.unsqueeze(1).expand(-1, self.n_layers, -1)

        if self.truncation_cutoff > 0:
            w_embs_trunc = self.w_mean.lerp(w_embs, self.truncation_psi)
            cutoff = torch.arange(self.n_layers) < self.truncation_cutoff
            cutoff = cutoff.view(1, self.n_layers, 1).to(w_embs.device)
            w_embs = torch.where(cutoff, w_embs_trunc, w_embs)

        return w_embs


class GeneratorBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        config: Config,
        first_block: bool = False,
        upsample: bool = False,
    ):
        super().__init__()
        if first_block:
            self.conv = nn.Identity()
        elif upsample:
            self.conv = up_conv_blur(in_dim, out_dim, 3, blur_size=config.blur_size)
        else:
            self.conv = conv3x3(in_dim, out_dim)
        self.noise_weight = nn.Parameter(torch.zeros(1, out_dim, 1, 1))  # B in paper
        self.act = config.act()
        self.norm = nn.InstanceNorm2d(out_dim)
        self.style_weight = nn.Linear(config.w_dim, out_dim)  # A in paper
        self.style_bias = nn.Linear(config.w_dim, out_dim)

    def forward(self, imgs: Tensor, w_embs: Tensor, noise: Optional[Tensor] = None):
        imgs = self.conv(imgs)
        b, c, h, w = imgs.shape

        noise = noise or torch.randn(b, 1, h, w, device=imgs.device)
        imgs = self.act(imgs + noise * self.noise_weight)

        style_weight = self.style_weight(w_embs).view(b, -1, 1, 1) + 1
        style_bias = self.style_bias(w_embs).view(b, -1, 1, 1)
        return self.norm(imgs) * style_weight + style_bias


class Generator(nn.Module):
    def __init__(self, config: Optional[Config] = None, **kwargs):
        config = config or Config()
        config = replace(config, **kwargs)
        assert config.img_size > 4 and math.log2(config.img_size).is_integer()
        super().__init__()
        self.mapping_network = MappingNetwork(config)

        map_size = config.smallest_map_size
        in_depth = config.input_depth
        self.learned_input = nn.Parameter(torch.empty(1, in_depth, map_size, map_size))

        block = partial(GeneratorBlock, config=config)
        depth = config.base_depth * config.img_size // map_size
        out_depth = min(depth, config.max_depth)

        self.layers = nn.ModuleList()
        self.layers.append(block(in_depth, in_depth, first_block=True))
        self.layers.append(block(in_depth, out_depth))
        in_depth = out_depth
        depth //= 2

        while map_size < config.img_size:
            out_depth = min(depth, config.max_depth)
            self.layers.append(block(in_depth, out_depth, upsample=True))
            self.layers.append(block(out_depth, out_depth))
            in_depth = out_depth
            depth //= 2
            map_size *= 2

        self.out_conv = conv1x1(in_depth, config.img_depth)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.learned_input)
        self.layers.apply(init_weights)

    def forward(self, z_embs: Tensor):
        w_embs = self.mapping_network(z_embs)  # (batch_size, n_layers, w_dim)
        x = self.learned_input.expand(z_embs.size(0), -1, -1, -1)
        for i, layer in enumerate(self.layers):
            x = layer(x, w_embs[:, i])
        return self.out_conv(x)
