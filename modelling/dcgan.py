# DCGAN - https://arxiv.org/abs/1701.07875
# Discriminator will downsample until the feature map is 4x4, then flatten and matmul
# Generator will matmul, reshape to 4x4, then upsample until the desired image size is obtained
#
# Code references:
# https://github.com/soumith/dcgan.torch/
# https://github.com/martinarjovsky/WassersteinGAN/

import math

from torch import nn


class DCGANDiscriminator(nn.Sequential):
    def __init__(self, img_channels: int = 3, img_size: int = 64, base_dim: int = 64) -> None:
        super().__init__()
        in_ch = img_channels

        # strided conv until 4x4
        for i in range(int(math.log2(img_size / 4))):
            out_ch = base_dim * 2**i
            new_layers = [
                nn.Conv2d(in_ch, out_ch, 4, 2, 1),
                nn.BatchNorm2d(out_ch, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            self.extend(new_layers)
            in_ch = out_ch

        # flatten and matmul
        self.extend([nn.Conv2d(in_ch, 1, 4), nn.Flatten(0)])
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)


class DCGANGenerator(nn.Sequential):
    def __init__(self, img_channels: int = 3, img_size: int = 64, base_dim: int = 64, z_dim: int = 128) -> None:
        out_ch = base_dim * img_size // 8

        # matmul and reshape to 4x4
        super().__init__(
            nn.Unflatten(-1, (z_dim, 1, 1)),
            nn.ConvTranspose2d(z_dim, out_ch, 4),
            nn.BatchNorm2d(out_ch, track_running_stats=False),
            nn.ReLU(inplace=True),
        )
        in_ch = out_ch

        # conv transpose until reaching image size / 2
        for _ in range(int(math.log2(img_size / 4)) - 1):
            out_ch = in_ch // 2
            new_layers = [
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                nn.BatchNorm2d(out_ch, track_running_stats=False),
                nn.ReLU(inplace=True),
            ]
            self.extend(new_layers)
            in_ch = out_ch

        # last layer: no bn and use tanh activation
        self.extend([nn.ConvTranspose2d(in_ch, img_channels, 4, 2, 1), nn.Tanh()])
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
