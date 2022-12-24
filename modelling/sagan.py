# SA-GAN - https://arxiv.org/pdf/1805.08318
# self-attention in SA-GAN is mostly identical to multi-head attention in ViT
# - single head (instead of multi-head)
# - no scaling factor 1/√dk
# - q, k, v have bottlenecked embedding dimension
# - k and v have reduced spatial resolution
#
# Code reference:
# https://github.com/brain-research/self-attention-gan
# https://github.com/ajbrock/BigGAN-PyTorch

from functools import partial
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import _Act, conv_norm_act


class SelfAttentionConv(nn.Module):
    def __init__(
        self,
        in_dim: int,
        qk_ratio: int = 8,
        v_ratio: int = 2,
    ):
        super().__init__()
        self.qkv_sizes = (in_dim // qk_ratio,) * 2 + (in_dim // v_ratio,)
        self.qkv_conv = nn.Conv2d(in_dim, sum(self.qkv_sizes), 1, bias=False)
        self.out_conv = nn.Conv2d(self.qkv_sizes[2], in_dim, 1, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, imgs: Tensor):
        b, c, h, w = imgs.shape
        q, k, v = torch.split(self.qkv_conv(imgs), self.qkv_sizes, dim=1)
        k, v = map(partial(F.max_pool2d, kernel_size=2), (k, v))
        q, k, v = map(partial(torch.flatten, start_dim=2), (q, k, v))

        # NOTE: ((Q K^T) V)^T = V^T (Q^T K)^T
        out = v @ (q.transpose(1, 2) @ k).softmax(dim=2).transpose(1, 2)
        return imgs + self.out_conv(out.view(b, -1, h, w)) * self.scale


class GeneratorStage(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, act: _Act = partial(nn.ReLU, True)):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            act(),
            nn.Upsample(scale_factor=2.0),
            nn.Conv2d(in_dim, out_dim, 3, padding=1, bias=False),
            conv_norm_act(out_dim, out_dim, 3, padding=1, order=["norm", "act", "conv"], act=act),
        )
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2.0),
            nn.Conv2d(in_dim, out_dim, 1),
        )

    def forward(self, imgs: Tensor):
        return self.layers(imgs) + self.shortcut(imgs)


class Generator(nn.Module):
    def __init__(
        self,
        img_size: int,
        img_depth: int,
        z_dim: int,
        smallest_map_size: int = 4,
        base_depth: int = 32,
        self_attention_sizes: Optional[List[int]] = None,
        act: _Act = partial(nn.ReLU, True),
    ):
        if self_attention_sizes is None:
            self_attention_sizes = [32]
        super().__init__()
        stage = partial(GeneratorStage, act=act)
        depth = base_depth * img_size // smallest_map_size // 2

        self.layers = nn.Sequential()
        self.layers.append(nn.ConvTranspose2d(z_dim, depth, smallest_map_size))
        in_depth = depth
        while smallest_map_size < img_size:
            self.layers.append(stage(in_depth, depth))
            in_depth = depth
            depth //= 2
            smallest_map_size *= 2

            if smallest_map_size in self_attention_sizes:
                self.layers.append(SelfAttentionConv(in_depth))

        self.layers.append(conv_norm_act(in_depth, img_depth, 3, padding=1, order=["norm", "act", "conv"], act=act))
        self.layers.append(nn.Tanh())

    def forward(self, z_embs: Tensor, ys: Optional[Tensor] = None):
        return self.layers(z_embs[:, :, None, None])
