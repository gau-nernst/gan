# Progressive GAN - https://arxiv.org/pdf/1710.10196
# See Table 2 for detailed model architecture
# Modifications:
# - Generator and Discriminator have skip-connections, introducted in StyleGAN2.
# - Residual connection: use pre-norm/pre-activation. Apply norm before activation.
# - PixelNorm in Generator is replaced with LayerNorm2d.
# - EqualizedLR is not implemented.
#
# Code reference:
# https://github.com/tkarras/progressive_growing_of_gans

import math

import torch
from torch import Tensor, nn


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        out = x.flatten(-2).transpose(-1, -2)
        out = super().forward(out)
        out = out.transpose(-1, -2).unflatten(-1, x.shape[-2:])
        return out


class MinibatchStdDev(nn.Module):
    def __init__(self, group_size: int = 4) -> None:
        super().__init__()
        self.group_size = group_size

    def forward(self, imgs: Tensor) -> Tensor:
        N, C, H, W = imgs.shape
        std = self.std(imgs.view(self.group_size, -1, C, H, W))
        std = std.mean([1, 2, 3], keepdim=True)
        std = std.repeat(self.group_size, 1, 1, 1).expand(N, 1, H, W)
        return torch.cat([imgs, std], dim=1)

    # torch.std() might return 0. when doing gradient penalty, this will result in division by zero in backward pass.
    # thus, we add eps=1e-6 before performing square root.
    @staticmethod
    def std(x: Tensor) -> Tensor:
        return (x - x.mean(0)).square().sum(0).div(x.shape[0]).add(1e-6).sqrt()


# this is identical SA-GAN, except activation function.
class ProgressiveGanDiscriminatorBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, residual: bool = False) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_dim, in_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_dim, out_dim, 3, 1, 1),
            nn.AvgPool2d(2),
        )
        if residual:
            self.shortcut = nn.Sequential(nn.AvgPool2d(2), nn.Conv2d(in_dim, out_dim, 1))
            self.scale = nn.Parameter(torch.full((out_dim, 1, 1), 1e-4))
        else:
            self.shortcut = None

    def forward(self, x: Tensor) -> Tensor:
        out = self.layers(x)
        if self.shortcut is not None:
            out = self.shortcut(x) + out
        return out


class ProgressiveGanDiscriminator(nn.Sequential):
    def __init__(self, img_size: int, img_channels: int = 3, base_dim: int = 16, residual: bool = False) -> None:
        super().__init__()
        depth = int(math.log2(img_size // 4))
        self.append(nn.Conv2d(img_channels, base_dim, 1))
        in_ch = base_dim

        for _ in range(depth):
            out_ch = min(in_ch * 2, 512)
            self.append(ProgressiveGanDiscriminatorBlock(in_ch, out_ch, residual=residual))
            in_ch = out_ch

        self.append(
            nn.Sequential(
                MinibatchStdDev(),
                nn.Conv2d(in_ch + 1, in_ch, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_ch, 1, 4),
                nn.Flatten(0),
            )
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)


class ProgressiveGanGeneratorBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, residual: bool = False) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            LayerNorm2d(in_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2.0),
            nn.Conv2d(in_dim, out_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_dim, out_dim, 3, 1, 1),
        )
        if residual:
            self.shortcut = nn.Sequential(nn.Conv2d(in_dim, out_dim, 1), nn.Upsample(scale_factor=2.0))
            self.scale = nn.Parameter(torch.full((out_dim, 1, 1), 1e-4))
        else:
            self.shortcut = None

    def forward(self, x: Tensor) -> Tensor:
        out = self.layers(x)
        if self.shortcut is not None:
            out = self.shortcut(x) + out * self.scale
        return out


class ProgressiveGanGenerator(nn.Sequential):
    def __init__(
        self, img_size: int, img_channels: int = 3, z_dim: int = 128, base_dim: int = 16, residual: bool = False
    ) -> None:
        super().__init__()
        out_ch = min(base_dim * img_size // 4, 512)
        self.append(
            nn.Sequential(
                nn.LayerNorm(z_dim),
                nn.Unflatten(-1, (-1, 1, 1)),
                nn.ConvTranspose2d(z_dim, out_ch, 4),
            )
        )
        in_ch = out_ch

        depth = int(math.log2(img_size // 4))
        for i in range(depth):
            out_ch = min(base_dim * img_size // 4 // 2 ** (i + 1), 512)
            self.append(ProgressiveGanGeneratorBlock(in_ch, out_ch, residual=residual))
            in_ch = out_ch

        self.append(
            nn.Sequential(
                LayerNorm2d(in_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_ch, img_channels, 1),
            )
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)


def init_weights(module: nn.Module):
    if isinstance(module, (nn.modules.conv._ConvNd, nn.Linear)):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
