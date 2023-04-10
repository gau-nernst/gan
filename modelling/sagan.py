# SA-GAN - https://arxiv.org/pdf/1805.08318
# self-attention in SA-GAN is mostly identical to multi-head attention in ViT
# - single head (instead of multi-head)
# - no scaling factor 1/√dk
# - q, k, v have bottlenecked embedding dimension
# - k and v have reduced spatial resolution
#
# For conditional generation
# - Discriminator: Projection Discrinator - https://arxiv.org/abs/1802.05637
# - Generator: Conditional Batch Norm - https://arxiv.org/abs/1707.00683
#
# Code reference:
# https://github.com/brain-research/self-attention-gan
# https://github.com/ajbrock/BigGAN-PyTorch

from functools import partial
from typing import Callable, List, Optional

import torch
from torch import Tensor, nn

from .base import _Act, conv1x1, conv3x3


_Layer = Callable[[int, int], nn.Module]


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, dim: int, y_dim: Optional[int] = None, y_layer_factory: _Layer = nn.Embedding):
        super().__init__()
        self.bn = nn.BatchNorm2d(dim, affine=y_dim is None, track_running_stats=False)
        self.weight = y_layer_factory(y_dim, dim) if y_dim is not None else None
        self.bias = y_layer_factory(y_dim, dim) if y_dim is not None else None

    def forward(self, imgs: Tensor, ys: Tensor):
        imgs = self.bn(imgs)
        if self.weight is not None and self.bias is not None:
            b = imgs.shape[0]
            weight = self.weight(ys).view(b, -1, 1, 1)
            bias = self.bias(ys).view(b, -1, 1, 1)
            imgs = imgs * weight + bias
        return imgs


class SelfAttentionConv2d(nn.Module):
    def __init__(self, in_dim: int, qk_ratio: int = 8, v_ratio: int = 2):
        assert in_dim % qk_ratio == 0 and in_dim % v_ratio == 0
        super().__init__()
        self.qkv_sizes = [in_dim // qk_ratio] * 2 + [in_dim // v_ratio]
        self.qkv_conv = conv1x1(in_dim, sum(self.qkv_sizes), bias=False)
        self.out_conv = conv1x1(self.qkv_sizes[2], in_dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.0))
        self.max_pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten(start_dim=2)

    def forward(self, imgs: Tensor):
        b, _, h, w = imgs.shape
        q, k, v = self.qkv_conv(imgs).split(self.qkv_sizes, dim=1)
        k, v = map(self.max_pool, (k, v))
        q, k, v = map(self.flatten, (q, k, v))

        # NOTE: (softmax(Q @ K^T) @ V)^T = V^T @ softmax(Q K^T)^T
        attn = q.transpose(1, 2).bmm(k).softmax(dim=2)  # (hw, c) @ (c, hw/4) -> (hw, hw/4)
        out = v.bmm(attn.transpose(1, 2))  # (c, hw/4) @ (hw/4, hw) -> (c, hw)
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
        # avg_pool + 1x1 conv is equivalent to 1x1 conv + avg_pool, but the former is faster
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
        y_dim: Optional[int] = None,
        smallest_map_size: int = 4,
        base_depth: int = 32,
        self_attention_sizes: Optional[List[int]] = None,
        y_layer_factory: _Layer = nn.Embedding,
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

        self.y_layer = y_layer_factory(y_dim, base_depth) if y_dim is not None else None
        self.out_layer = conv1x1(base_depth, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)

    def forward(self, imgs: Tensor, ys: Optional[Tensor] = None):
        embs = self.layers(imgs)
        logits = self.out_layer(embs).view(-1)
        if self.y_layer is not None:  # projection Discriminator
            b = imgs.shape[0]
            y_logits = torch.bmm(embs.view(b, 1, -1), self.y_layer(ys).view(b, -1, 1))
            logits = logits + y_logits.view(b)
        return logits


class GeneratorStage(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        y_dim: Optional[int] = None,
        y_layer_factory: _Layer = nn.Embedding,
        act: _Act = partial(nn.ReLU, True),
    ):
        super().__init__()
        norm = partial(ConditionalBatchNorm2d, y_dim=y_dim, y_layer_factory=y_layer_factory)
        self.layers = nn.ModuleList()

        self.layers.append(norm(in_dim))
        self.layers.append(act())
        self.layers.append(nn.Upsample(scale_factor=2.0))
        self.layers.append(conv3x3(in_dim, out_dim, bias=False))

        self.layers.append(norm(out_dim))
        self.layers.append(act())
        self.layers.append(conv3x3(out_dim, out_dim))

        # 1x1 conv + upsample is equivalent to upsample + 1x1 conv, but the former is faster
        self.shortcut = nn.Sequential(conv1x1(in_dim, out_dim), nn.Upsample(scale_factor=2.0))

    def forward(self, imgs: Tensor, ys: Optional[Tensor] = None):
        shortcut = self.shortcut(imgs)
        for layer in self.layers:
            imgs = layer(imgs, ys) if isinstance(layer, ConditionalBatchNorm2d) else layer(imgs)
        return imgs + shortcut


class Generator(nn.Module):
    def __init__(
        self,
        img_size: int,
        img_depth: int,
        z_dim: int,
        y_dim: Optional[int] = None,
        smallest_map_size: int = 4,
        base_depth: int = 32,
        self_attention_sizes: Optional[List[int]] = None,
        y_layer_factory: _Layer = nn.Embedding,
        act: _Act = partial(nn.ReLU, True),
    ):
        self_attention_sizes = self_attention_sizes or [32]
        super().__init__()
        stage = partial(GeneratorStage, y_dim=y_dim, y_layer_factory=y_layer_factory, act=act)
        depth = base_depth * img_size // smallest_map_size // 2

        self.layers = nn.ModuleList()
        self.layers.append(nn.ConvTranspose2d(z_dim, depth, smallest_map_size))
        in_depth = depth
        while smallest_map_size < img_size:
            self.layers.append(stage(in_depth, depth))
            in_depth = depth
            depth //= 2
            smallest_map_size *= 2

            if smallest_map_size in self_attention_sizes:
                self.layers.append(SelfAttentionConv2d(in_depth))

        self.layers.append(nn.BatchNorm2d(in_depth))
        self.layers.append(act())
        self.layers.append(conv3x3(in_depth, img_depth))
        self.layers.append(nn.Tanh())

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)

    def forward(self, z_embs: Tensor, ys: Optional[Tensor] = None):
        out = z_embs.view(*z_embs.shape, 1, 1)
        for layer in self.layers:
            out = layer(out, ys) if isinstance(layer, GeneratorStage) else layer(out)
        return out


def init_weights(module: nn.Module):
    if isinstance(module, nn.modules.conv._ConvNd):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
