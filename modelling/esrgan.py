# ESRGAN - https://arxiv.org/abs/1809.00219
#
# Code reference:
# https://github.com/xinntao/ESRGAN

import torch
from torch import Tensor, nn


# inspired by DenseNet - https://arxiv.org/abs/1608.06993
class DenseBlock(nn.ModuleList):
    def __init__(self, dim: int, growth_rate: int = 32, depth: int = 5):
        super().__init__()
        for i in range(depth - 1):
            self.append(
                nn.Sequential(
                    nn.Conv2d(dim + growth_rate * i, growth_rate, 3, 1, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
        self.append(nn.Conv2d(dim + growth_rate * (depth - 1), dim, 3, 1, 1))

    def forward(self, imgs: Tensor) -> Tensor:
        out = [imgs]
        for layer in self:
            out.append(layer(torch.cat(out, 1)))
        return out[-1]


class RRDBlock(nn.Module):
    def __init__(self, channels: int, n_blocks: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(DenseBlock(channels))

    def forward(self, imgs):
        for block in self.blocks:
            imgs = imgs + block(imgs)
        return imgs


class ESRGANGenerator(nn.Module):
    def __init__(self, img_channels: int = 3, base_dim: int = 64, n_blocks: int = 8, upsample: int = 2) -> None:
        super().__init__()
        self.input_layer = nn.Conv2d(img_channels, base_dim, 3, 1, 1)

        self.trunk = nn.Sequential(*[RRDBlock(base_dim) for _ in range(n_blocks)])
        self.trunk.append(nn.Conv2d(base_dim, base_dim, 3, 1, 1))

        self.output_layer = nn.Sequential()
        for _ in range(upsample):
            self.output_layer.extend(
                [
                    nn.Upsample(scale_factor=2.0),
                    nn.ConvTranspose2d(base_dim, base_dim, 3, 1, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        self.output_layer.extend(
            [
                nn.Conv2d(base_dim, base_dim, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_dim, img_channels, 3, 1, 1),
            ]
        )

    def forward(self, imgs: Tensor) -> Tensor:
        out = self.input_layer(imgs)
        out = out + self.trunk(out)
        out = self.output_layer(out)
        return out
