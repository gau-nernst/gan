# SRGAN - https://arxiv.org/abs/1609.04802
#
# Original implementation uses Pixel Shuffle. It is equivalent to ConvTranspose
# https://arxiv.org/abs/1609.07009
# ConvTranspose is faster on CUDA most of the time. There is no reason to use Pixel Shuffle.

from torch import Tensor, nn

from .cyclegan import ResNetBlock


class SRResNet(nn.Module):
    def __init__(self, img_channels: int = 3, base_channels: int = 64, n_blocks: int = 16, upsample: int = 2):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Conv2d(img_channels, base_channels, 9, 1, 4), nn.PReLU())

        self.blocks = nn.Sequential()
        for _ in range(n_blocks):
            self.blocks.append(ResNetBlock(base_channels))
        self.blocks.append(
            nn.Sequential(
                nn.Conv2d(base_channels, base_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(base_channels),
            )
        )

        self.output_layer = nn.Sequential()
        for _ in range(upsample):
            self.output_layer.extend([nn.ConvTranspose2d(base_channels, base_channels, 6, 2, 2), nn.PReLU()])
        self.output_layer.append(nn.Conv2d(base_channels, img_channels, 9, 1, 4))

    def forward(self, x: Tensor) -> Tensor:
        out = self.input_layer(x)
        out = out + self.blocks(out)
        out = self.output_layer(out)
        return out


class SRGANDiscriminator(nn.Sequential):
    def __init__(self, img_channels: int = 3, base_dim: int = 64):
        super().__init__()

        def get_nc(idx: int):
            return base_dim * 2 ** (idx // 2)

        self.extend([nn.Conv2d(img_channels, base_dim, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)])

        for i in range(1, 8):
            self.append(
                nn.Sequential(
                    nn.Conv2d(get_nc(i - 1), get_nc(i), 3, i % 2 + 1, 1),
                    nn.BatchNorm2d(get_nc(i)),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )

        self.extend([nn.Conv2d(get_nc(7), get_nc(8), 1), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(get_nc(8), 1, 1)])
