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
        imgs = imgs.float()  # always compute in fp32
        N, C, H, W = imgs.shape
        std = imgs.view(self.group_size, -1, C, H, W).std(0, unbiased=False)
        std = std.mean([1, 2, 3], keepdim=True)
        std = std.repeat(self.group_size, 1, 1, 1).expand(N, 1, H, W)
        return torch.cat([imgs, std], dim=1)


# this is identical SA-GAN, except activation function.
class ProgressiveGanDiscriminatorBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_dim, in_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_dim, out_dim, 3, 1, 1),
            nn.AvgPool2d(2),
        )
        self.shortcut = nn.Sequential(nn.AvgPool2d(2), nn.Conv2d(in_dim, out_dim, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.residual(x) + self.shortcut(x)


class ProgressiveGanDiscriminator(nn.Sequential):
    def __init__(self, img_size: int, img_channels: int = 3, base_dim: int = 16) -> None:
        super().__init__()
        depth = int(math.log2(img_size // 4))
        self.append(nn.Conv2d(img_channels, base_dim, 1))
        in_ch = base_dim

        for _ in range(depth):
            out_ch = min(in_ch * 2, 512)
            self.append(ProgressiveGanDiscriminatorBlock(in_ch, out_ch))
            in_ch = out_ch

        out_ch = min(in_ch * 2, 512)
        self.append(
            nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                MinibatchStdDev(),
                nn.Conv2d(in_ch + 1, in_ch, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_ch, out_ch, 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, 1, 1),
                nn.Flatten(0),
            )
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)


class ProgressiveGanGeneratorBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            LayerNorm2d(in_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2.0),
            nn.Conv2d(in_dim, out_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_dim, out_dim, 3, 1, 1),
        )
        self.shortcut = nn.Sequential(nn.Conv2d(in_dim, out_dim, 1), nn.Upsample(scale_factor=2.0))

    def forward(self, x: Tensor) -> Tensor:
        return self.residual(x) + self.shortcut(x)


class ProgressiveGanGenerator(nn.Sequential):
    def __init__(self, img_size: int, img_channels: int = 3, z_dim: int = 128, base_dim: int = 16) -> None:
        super().__init__()
        out_ch = min(base_dim * img_size // 4, 512)
        self.append(
            nn.Sequential(
                nn.LayerNorm(z_dim),
                nn.Unflatten(-1, (-1, 1, 1)),
                nn.ConvTranspose2d(z_dim, out_ch, 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            )
        )
        in_ch = out_ch

        depth = int(math.log2(img_size // 4))
        for i in range(depth):
            out_ch = min(base_dim * img_size // 4 // 2 ** (i + 1), 512)
            self.append(ProgressiveGanGeneratorBlock(in_ch, out_ch))
            in_ch = out_ch

        self.append(
            nn.Sequential(
                LayerNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_ch, img_channels, 1),
            )
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)


def init_weights(module: nn.Module):
    if isinstance(module, (nn.modules.conv._ConvNd, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, a=0.2)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
