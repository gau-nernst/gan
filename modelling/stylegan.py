import math
from functools import partial
from typing import Callable, List, Optional

import torch
from torch import Tensor, nn

from .base import _Act, _Norm


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

        nn.init.zeros_(self.noise_weight)
        nn.init.kaiming_normal_(self.style_linear.weight)
        nn.init.ones_(self.style_linear.bias[:out_dim])  # scale
        nn.init.zeros_(self.style_linear.bias[out_dim:])  # bias

    def forward(self, imgs: Tensor, w_embs: Tensor):
        imgs = self.conv(imgs)
        b, c, h, w = imgs.shape

        noise = torch.randn(b, 1, h, w, device=imgs.device)
        imgs = self.act(imgs + noise * self.noise_weight)

        style = self.style_linear(w_embs).reshape(b, 2, c, 1, 1)
        return self.norm(imgs) * style[:, 0] + style[:, 1]


class Generator(nn.Module):
    def __init__(
        self,
        img_size: int,
        img_depth: int,
        z_dim: int,
        w_dim: int = 512,
        learned_input_depth: int = 512,
        learned_input_size: int = 4,
        base_depth: int = 32,
        max_depth: int = 512,
        act: _Act = partial(nn.ReLU, inplace=True),
    ):
        assert img_size > 4 and math.log2(img_size).is_integer()
        super().__init__()
        self.mapping_network = self.mlp(z_dim, [w_dim] * 8, None, act)

        self.learned_input = nn.Parameter(torch.empty(1, learned_input_depth, learned_input_size, learned_input_size))

        conv3x3 = partial(nn.Conv2d, kernel_size=3, padding=1)
        up_conv = partial(nn.ConvTranspose2d, kernel_size=4, stride=2, padding=1)
        block = partial(GeneratorBlock(w_dim=w_dim, act=act))
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
    def mlp(in_dim: int, dim_list: List[int], norm: _Norm, act: _Act):
        # apply norm+act at all layers, including output layer
        network = nn.Sequential()
        for dim in dim_list:
            network.append(nn.Linear(in_dim, dim, bias=norm is None))
            network.append(norm(dim) if norm is not None else nn.Identity())
            network.append(act() if act is not None else nn.Identity())
            in_dim = dim
        return network


class Discriminator(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, imgs: Tensor):
        pass
