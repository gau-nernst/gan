# StyleGAN - https://arxiv.org/abs/1812.04948
# Not implemented features
# - BlurConv (https://arxiv.org/abs/1904.11486): helps to reduce anti-aliasing
# - Progressive growing and Equalized learning rate (https://arxiv.org/abs/1710.10196)
#
# Code reference:
# https://github.com/NVlabs/stylegan

import math
from functools import partial
from typing import Callable, List, Optional

import torch
from torch import Tensor, nn

from .base import _Act, conv_norm_act


class GeneratorBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        w_dim: int,
        conv: Optional[Callable[..., nn.Module]],
        act: _Act,
    ):
        super().__init__()
        self.conv = conv(in_dim, out_dim)
        self.noise_weight = nn.Parameter(torch.empty(1, out_dim, 1, 1))  # B in paper
        self.act = act()
        self.norm = nn.InstanceNorm2d(out_dim)
        self.style_linear = nn.Linear(w_dim, out_dim * 2)  # A in paper

        if isinstance(self.conv, nn.modules.conv._ConvNd):
            nn.init.kaiming_normal_(self.conv.weight)
            nn.init.zeros_(self.conv.bias)
        nn.init.zeros_(self.noise_weight)
        nn.init.kaiming_normal_(self.style_linear.weight)
        nn.init.ones_(self.style_linear.bias[:out_dim])  # weight
        nn.init.zeros_(self.style_linear.bias[out_dim:])  # bias

    def forward(self, imgs: Tensor, w_embs: Tensor):
        imgs = self.conv(imgs)
        b, c, h, w = imgs.shape

        noise = torch.randn(b, 1, h, w, device=imgs.device)
        imgs = self.act(imgs + noise * self.noise_weight)

        style_w, style_b = torch.chunk(self.style_linear(w_embs), 2, dim=1)
        return self.norm(imgs) * style_w + style_b


class Generator(nn.Module):
    def __init__(
        self,
        img_size: int,
        img_depth: int = 3,
        z_dim: int = 512,
        w_dim: int = 512,
        learned_input_depth: int = 512,
        learned_input_size: int = 4,
        base_depth: int = 32,
        max_depth: int = 512,
        act: _Act = partial(nn.LeakyReLU, 0.2, True),
    ):
        assert img_size > 4 and math.log2(img_size).is_integer()
        super().__init__()
        self.mapping_network = self.mlp(z_dim, [w_dim] * 8, act)

        self.learned_input = nn.Parameter(torch.empty(1, learned_input_depth, learned_input_size, learned_input_size))

        conv3x3 = partial(nn.Conv2d, kernel_size=3, padding=1)
        up_conv = partial(nn.ConvTranspose2d, kernel_size=4, stride=2, padding=1)
        block = partial(GeneratorBlock, w_dim=w_dim, act=act)
        depth = base_depth * img_size // learned_input_size

        in_depth = learned_input_depth
        in_size = learned_input_size
        _depth = min(depth, max_depth)

        self.stages = nn.ModuleList()
        first_stage = nn.ModuleList()
        first_stage.append(block(in_depth, in_depth, conv=nn.Identity))
        first_stage.append(block(in_depth, _depth, conv=conv3x3))
        self.stages.append(first_stage)
        in_depth, depth = _depth, depth // 2

        while in_size < img_size:
            _depth = min(depth, max_depth)
            stage = nn.ModuleList()
            stage.append(block(in_depth, _depth, conv=up_conv))
            stage.append(block(_depth, _depth, conv=conv3x3))
            self.stages.append(stage)
            in_depth, depth, in_size = _depth, depth // 2, in_size * 2

        self.out_conv = nn.Conv2d(in_depth, img_depth, 1)

        nn.init.ones_(self.learned_input)

    def forward(self, z_embs: Tensor, ys: Optional[Tensor] = None):
        b = z_embs.size(0)
        w_embs = self.mapping_network(z_embs)

        x = self.learned_input.repeat(b, 1, 1, 1)
        for stage in self.stages:
            for block in stage:
                x = block(x, w_embs)
        return self.out_conv(x)

    @staticmethod
    def mlp(in_dim: int, dim_list: List[int], act: _Act):
        # apply act at all layers, including output layer
        network = nn.Sequential()
        for dim in dim_list:
            network.append(nn.Linear(in_dim, dim))
            network.append(act() if act is not None else nn.Identity())
            in_dim = dim
        return network


class Discriminator(nn.Module):
    def __init__(
        self,
        img_size,
        img_depth: int = 3,
        base_depth: int = 32,
        max_depth: int = 512,
        smallest_map_size: int = 4,
        act: _Act = partial(nn.LeakyReLU, 0.2, True),
    ):
        assert img_size > 4 and math.log2(img_size).is_integer()
        super().__init__()
        kwargs = dict(norm=None, act=act)
        conv3x3 = partial(conv_norm_act, kernel_size=3, padding=1, **kwargs)
        conv1x1 = partial(conv_norm_act, kernel_size=1, **kwargs)

        self.layers = nn.Sequential()
        self.layers.append(conv1x1(img_depth, base_depth))

        while img_size > smallest_map_size:
            self.layers.append(conv3x3(base_depth, base_depth))
            self.layers.append(conv3x3(base_depth, min(base_depth * 2, max_depth), stride=2))
            img_size //= 2
            base_depth = min(base_depth * 2, max_depth)

        # TODO: minibatch_stddev
        self.layers.append(conv3x3(base_depth, base_depth))
        self.layers.append(conv_norm_act(base_depth, base_depth, smallest_map_size, **kwargs))
        self.layers.append(conv1x1(base_depth, 1))

        for module in self.layers.modules():
            if isinstance(module, nn.modules.conv._ConvNd):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, imgs: Tensor, ys: Optional[Tensor] = None):
        return self.layers(imgs)
