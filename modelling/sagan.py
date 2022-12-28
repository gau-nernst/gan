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
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import _Act, _Norm, conv1x1, conv3x3, conv_norm_act


class ConditionalBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features: int, y_features: int, layer: Callable[[int, int], nn.Module] = nn.Embedding):
        super().__init__(num_features, affine=False)
        self.weight_proj = layer(y_features, num_features)
        self.bias_proj = layer(y_features, num_features)

    def forward(self, imgs: Tensor, ys: Tensor):
        imgs = super().forward(imgs)()
        weight = self.weight_proj(ys).view(-1, self.num_features, 1, 1)
        bias = self.bias_proj(ys).view(-1, self.num_features, 1, 1)
        return imgs * weight + bias


class SelfAttentionConv2d(nn.Module):
    def __init__(
        self,
        in_dim: int,
        qk_ratio: int = 8,
        v_ratio: int = 2,
    ):
        super().__init__()
        self.qkv_sizes = (in_dim // qk_ratio,) * 2 + (in_dim // v_ratio,)
        self.qkv_conv = conv1x1(in_dim, sum(self.qkv_sizes), bias=False)
        self.out_conv = conv1x1(self.qkv_sizes[2], in_dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, imgs: Tensor):
        b, c, h, w = imgs.shape
        q, k, v = torch.split(self.qkv_conv(imgs), self.qkv_sizes, dim=1)
        k, v = map(partial(F.max_pool2d, kernel_size=2), (k, v))
        q, k, v = map(partial(torch.flatten, start_dim=2), (q, k, v))

        # NOTE: ((Q K^T) V)^T = V^T (Q^T K)^T
        out = v @ (q.transpose(1, 2) @ k).softmax(dim=2).transpose(1, 2)
        return imgs + self.out_conv(out.view(b, -1, h, w)) * self.scale


class DiscriminatorBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        first_block: bool = False,
        downsample: bool = True,
        act: _Act = partial(nn.ReLU, True),
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Identity() if first_block else act(),
            conv3x3(in_dim, out_dim),
            act(),
            conv3x3(out_dim, out_dim),
            nn.AvgPool2d(2) if downsample else nn.Identity(),
        )
        # NOTE: avg_pool + 1x1 conv is equivalent to 1x1 conv + avg_pool, but the former should be faster
        if downsample:
            self.shortcut = nn.Sequential(nn.AvgPool2d(2), conv1x1(in_dim, out_dim))
        elif out_dim != in_dim:
            self.shortcut = conv1x1(in_dim, out_dim)
        else:
            self.shortcut = nn.Identity()

    def forward(self, imgs: Tensor):
        return self.layers(imgs) + self.shortcut(imgs)


class Discriminator(nn.Module):
    def __init__(
        self,
        img_size: int,
        img_depth: int,
        smallest_map_size: int = 4,
        base_depth: int = 32,
        self_attention_sizes: Optional[List[int]] = None,
        act: _Act = partial(nn.ReLU, True),
    ):
        if self_attention_sizes is None:
            self_attention_sizes = [32]
        super().__init__()
        block = partial(DiscriminatorBlock, act=act)
        self.layers = nn.Sequential()
        self.layers.append(block(img_depth, base_depth, first_block=True))
        img_size //= 2

        while img_size > smallest_map_size:
            self.layers.append(block(base_depth, base_depth * 2))
            base_depth *= 2
            img_size //= 2

            if img_size in self_attention_sizes:
                self.layers.append(SelfAttentionConv2d(base_depth))

        self.layers.append(block(base_depth, base_depth, downsample=False))
        self.layers.append(act())
        self.layers.append(nn.AvgPool2d(4))  # official implementation uses sum
        self.layers.append(conv1x1(base_depth, 1))

    def forward(self, imgs: Tensor, ys: Optional[Tensor] = None):
        return self.layers(imgs)


class GeneratorStage(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        norm: _Norm = nn.BatchNorm2d,
        act: _Act = partial(nn.ReLU, True),
    ):
        super().__init__()
        layers = [
            norm(in_dim),
            act(),
            nn.Upsample(scale_factor=2.0),
            conv3x3(in_dim, out_dim, bias=False),
            norm(out_dim),
            act(),
            conv3x3(out_dim, out_dim),
        ]
        self.layers = nn.ModuleList(layers)
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2.0),
            conv1x1(in_dim, out_dim),
        )

    def forward(self, imgs: Tensor, ys: Optional[Tensor] = None):
        shortcut = self.shortcut(imgs)
        for layer in self.layers:
            if isinstance(layer, ConditionalBatchNorm2d):
                imgs = layer(imgs, ys)
            else:
                imgs = layer(imgs)
        return imgs + shortcut


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
                self.layers.append(SelfAttentionConv2d(in_depth))

        self.layers.append(conv_norm_act(in_depth, img_depth, 3, padding=1, order=["norm", "act", "conv"], act=act))
        self.layers.append(nn.Tanh())

    def forward(self, z_embs: Tensor, ys: Optional[Tensor] = None):
        return self.layers(z_embs[:, :, None, None])
