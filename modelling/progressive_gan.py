# Progressive GAN - https://arxiv.org/pdf/1710.10196
# See Table 2 for detailed model architecture
# Discriminator supports BlurConv (StyleGAN) and skip-connections (StyleGAN2)
# Generator has an option to use BlurConv
#
# Code reference:
# https://github.com/tkarras/progressive_growing_of_gans

import math

import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize


class PixelNorm(nn.Module):
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()  # always compute in fp32
        return x * x.square().mean(1, keepdim=True).add(self.eps).rsqrt()


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


class EqualizedLR(nn.Module):
    def __init__(self, weight: Tensor):
        super().__init__()
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        self.scale = 2**0.5 / fan_in**0.5  # use gain=sqrt(2) everywhere

    def forward(self, weight: Tensor):
        return weight * self.scale

    def extra_repr(self) -> str:
        return f"scale={self.scale}"


# with residual connection introduced in StyleGAN2. very similar to SA-GAN
class ProgressiveGanDiscriminatorBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_dim, out_dim, 3, 1, 1),
            nn.AvgPool2d(2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.shortcut = nn.Sequential(nn.AvgPool2d(2), nn.Conv2d(in_dim, out_dim, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.residual(x) + self.shortcut(x)


class ProgressiveGanDiscriminator(nn.Sequential):
    def __init__(self, img_size: int, img_channels: int = 3, base_dim: int = 16) -> None:
        super().__init__()
        depth = int(math.log2(img_size // 4))
        self.append(nn.Sequential(nn.Conv2d(img_channels, base_dim, 1), nn.LeakyReLU(0.2, inplace=True)))
        in_ch = base_dim

        for _ in range(depth):
            out_ch = min(in_ch * 2, 512)
            self.append(ProgressiveGanDiscriminatorBlock(in_ch, out_ch))
            in_ch = out_ch

        out_ch = min(in_ch * 2, 512)
        self.append(
            nn.Sequential(
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


class ProgressiveGanGenerator(nn.Sequential):
    def __init__(self, img_size: int, img_channels: int = 3, z_dim: int = 128, base_dim: int = 16) -> None:
        super().__init__()
        self.input_norm = PixelNorm()

        out_ch = min(base_dim * img_size // 4, 512)
        self.append(
            nn.Sequential(
                nn.Unflatten(-1, (-1, 1, 1)),
                nn.ConvTranspose2d(z_dim, out_ch, 4),
                nn.LeakyReLU(0.2, inplace=True),
                PixelNorm(),
                nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                PixelNorm(),
            )
        )
        in_ch = out_ch

        depth = int(math.log2(img_size // 4))
        for i in range(depth):
            out_ch = min(base_dim * img_size // 4 // 2 ** (i + 1), 512)
            self.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2.0),
                    nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                    PixelNorm(),
                    nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                    PixelNorm(),
                )
            )
            in_ch = out_ch

        self.append(nn.Conv2d(in_ch, img_channels, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)


def init_weights(module: nn.Module):
    if isinstance(module, (nn.modules.conv._ConvNd, nn.Linear)):
        nn.init.normal_(module.weight)
        parametrize.register_parametrization(module, "weight", EqualizedLR(module.weight))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
