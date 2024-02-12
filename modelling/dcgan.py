# DCGAN - https://arxiv.org/abs/1701.07875
# Discriminator will downsample until the feature map is 4x4, then flatten and matmul
# Generator will matmul, reshape to 4x4, then upsample until the desired image size is obtained
#
# Code references:
# https://github.com/soumith/dcgan.torch/
# https://github.com/martinarjovsky/WassersteinGAN/

import math

from torch import nn

from .common import get_norm


class DcGanDiscriminator(nn.Sequential):
    def __init__(self, img_channels: int = 3, img_size: int = 64, base_dim: int = 64, norm: str = "bn") -> None:
        # no bn in the first conv layer
        super().__init__(
            nn.Conv2d(img_channels, base_dim, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        in_ch = base_dim

        depth = int(math.log2(img_size / 4))  # downsample until 4x4
        for _ in range(depth - 1):
            self.append(nn.Conv2d(in_ch, in_ch * 2, 4, 2, 1))
            self.append(get_norm(in_ch * 2, norm))
            self.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch *= 2

        # flatten and matmul
        self.append(nn.Conv2d(in_ch, 1, 4))
        self.append(nn.Flatten(0))
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)


class DcGanGenerator(nn.Sequential):
    def __init__(
        self, img_channels: int = 3, img_size: int = 64, base_dim: int = 64, z_dim: int = 128, norm: str = "bn"
    ) -> None:
        depth = int(math.log2(img_size / 4))  # upsample from 4x4 to img_size
        out_ch = base_dim * 2 ** (depth - 1)

        # matmul and reshape to 4x4
        super().__init__(
            nn.Unflatten(-1, (z_dim, 1, 1)),
            nn.ConvTranspose2d(z_dim, out_ch, 4),
            get_norm(out_ch, norm),
            nn.ReLU(inplace=True),
        )
        in_ch = out_ch

        # conv transpose until reaching image size / 2
        for _ in range(depth - 1):
            self.append(nn.ConvTranspose2d(in_ch, in_ch // 2, 4, 2, 1))
            self.append(get_norm(in_ch // 2, norm))
            self.append(nn.ReLU(inplace=True))
            in_ch //= 2

        # last layer: no bn and use tanh activation
        self.append(nn.ConvTranspose2d(in_ch, img_channels, 4, 2, 1))
        self.append(nn.Tanh())
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)


def init_weights(module: nn.Module):
    if isinstance(module, nn.modules.conv._ConvNd):
        nn.init.normal_(module.weight, 0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.modules.batchnorm._BatchNorm):
        nn.init.normal_(module.weight, 1, 0.02)
        nn.init.zeros_(module.bias)
