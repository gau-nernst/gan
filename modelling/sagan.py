# SA-GAN - https://arxiv.org/pdf/1805.08318
# self-attention in SA-GAN is mostly identical to multi-head attention in ViT
# - single head (instead of multi-head)
# - no scaling factor 1/sqrt(dk)
# - q, k, v have reduced embedding dimension
# - k and v have reduced spatial resolution
#
# For conditional generation
# - Discriminator: Projection Discrinator - https://arxiv.org/abs/1802.05637
# - Generator: Conditional Batch Norm - https://arxiv.org/abs/1707.00683
#
# Code reference:
# https://github.com/brain-research/self-attention-gan
# https://github.com/ajbrock/BigGAN-PyTorch

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, dim: int, n_classes: int = 1, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Embedding(n_classes, dim)
        self.bias = nn.Embedding(n_classes, dim)
        self.eps = eps

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = F.batch_norm(x, None, None, training=True, eps=self.eps)
        weight = self.weight(y).unflatten(-1, (-1, 1, 1))
        bias = self.bias(y).unflatten(-1, (-1, 1, 1))
        return x * weight + bias


class SelfAttention2d(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Conv2d(in_dim, in_dim // 8, 1, bias=False)
        self.k_proj = nn.Conv2d(in_dim, in_dim // 8, 1, bias=False)
        self.v_proj = nn.Conv2d(in_dim, in_dim // 2, 1, bias=False)
        self.out_proj = nn.Conv2d(in_dim // 2, in_dim, 1, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.0))  # layer scale
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x: Tensor) -> Tensor:
        q = self.q_proj(x).flatten(-2).transpose(-1, -2)
        k = self.max_pool(self.k_proj(x)).flatten(-2).transpose(-1, -2)
        v = self.max_pool(self.v_proj(x)).flatten(-2).transpose(-1, -2)

        out = F.scaled_dot_product_attention(q, k, v)  # original code doesn't use 1/sqrt(dk)
        out = out.transpose(-1, -2).unflatten(-1, x.shape[-2:])
        return x + self.out_proj(out) * self.scale


class SAGANDiscriminatorBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, first_block: bool = False) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            nn.Identity() if first_block else nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, out_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 3, 1, 1),
            nn.AvgPool2d(2),
        )
        # avg_pool + 1x1 conv is equivalent to 1x1 conv + avg_pool, but the former is faster
        self.shortcut = nn.Sequential(nn.AvgPool2d(2), nn.Conv2d(in_dim, out_dim, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.residual(x) + self.shortcut(x)


class SAGANDiscriminator(nn.Module):
    def __init__(
        self,
        img_size: int,
        img_channels: int,
        n_classes: int = 1,
        base_depth: int = 32,
        self_attn_sizes: list[int] | None = None,
    ) -> None:
        if self_attn_sizes is None:
            self_attn_sizes = [32]
        super().__init__()
        self.blocks = nn.Sequential()
        self.blocks.append(SAGANDiscriminatorBlock(img_channels, base_depth, first_block=True))
        img_size //= 2

        if img_size in self_attn_sizes:
            self.blocks.append(SelfAttention2d(base_depth))

        while img_size > 4:
            self.blocks.append(SAGANDiscriminatorBlock(base_depth, base_depth * 2))
            base_depth *= 2
            img_size //= 2

            if img_size in self_attn_sizes:
                self.blocks.append(SelfAttention2d(base_depth))

        self.blocks.append(
            nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(base_depth, base_depth, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_depth, base_depth, 3, 1, 1),
                nn.ReLU(inplace=True),
            )
        )
        self.y_proj = nn.Embedding(n_classes, base_depth)
        self.bias = nn.Parameter(torch.zeros(base_depth))

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)

    def forward(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        if y is None:
            y = x.new_zeros(1, dtype=torch.long)
        embs = self.blocks(x).sum(dim=(-1, -2))
        return (embs * (self.y_proj(y) + self.bias)).sum(1)


class SAGANGeneratorStage(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_classes: int) -> None:
        super().__init__()
        residual = [
            ConditionalBatchNorm2d(in_dim, n_classes),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2.0),
            nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=False),
            ConditionalBatchNorm2d(out_dim, n_classes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 3, 1, 1),
        ]
        self.residual = nn.ModuleList(residual)

        # 1x1 conv + upsample is equivalent to upsample + 1x1 conv, but the former is faster
        self.shortcut = nn.Sequential(nn.Conv2d(in_dim, out_dim, 1), nn.Upsample(scale_factor=2.0))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        shortcut = self.shortcut(x)
        for layer in self.residual:
            x = layer(x, y) if isinstance(layer, ConditionalBatchNorm2d) else layer(x)
        return x + shortcut


class SAGANGenerator(nn.Module):
    def __init__(
        self,
        img_size: int,
        img_channels: int,
        z_dim: int,
        n_classes: int = 1,
        smallest_map_size: int = 4,
        base_depth: int = 32,
        self_attention_sizes: list[int] | None = None,
    ) -> None:
        self_attention_sizes = self_attention_sizes or [32]
        super().__init__()
        depth = base_depth * img_size // smallest_map_size // 2

        self.layers = nn.ModuleList()
        self.layers.append(nn.ConvTranspose2d(z_dim, depth, smallest_map_size))
        in_depth = depth
        while smallest_map_size < img_size:
            self.layers.append(SAGANGeneratorStage(in_depth, depth, n_classes))
            in_depth = depth
            depth //= 2
            smallest_map_size *= 2

            if smallest_map_size in self_attention_sizes:
                self.layers.append(SelfAttention2d(in_depth))

        self.layers.append(
            nn.Sequential(
                nn.BatchNorm2d(in_depth),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_depth, img_channels, 3, 1, 1),
                nn.Tanh(),
            )
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)

    def forward(self, z: Tensor, y: Tensor | None = None) -> Tensor:
        if y is None:
            y = z.new_zeros(1, dtype=torch.long)
        out = z.unflatten(-1, (-1, 1, 1))
        for layer in self.layers:
            out = layer(out, y) if isinstance(layer, SAGANGeneratorStage) else layer(out)
        return out


def init_weights(module: nn.Module):
    if isinstance(module, nn.modules.conv._ConvNd):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    # TODO: nn.Embedding
