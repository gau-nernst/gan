from dataclasses import dataclass
from functools import partial

from torch import Tensor, nn

from .base import _Act, _Norm, conv3x3


conv7x7 = partial(nn.Conv2d, kernel_size=7, padding=3)
upconv3x3 = partial(nn.ConvTranspose2d, kernel_size=3, stride=2, padding=1, output_padding=1)


@dataclass
class CycleGANConfig:
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 64
    n_blocks: int = 4
    downsample: int = 2
    norm: _Norm = nn.InstanceNorm2d
    act: _Act = partial(nn.ReLU, inplace=True)


# almost identical to torchvision.models.resnet.BasicBlock
class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, config: CycleGANConfig):
        super().__init__()
        self.main = nn.Sequential(
            conv3x3(in_channels, out_channels, bias=False),
            config.norm(out_channels),
            config.act(),
            conv3x3(out_channels, out_channels, bias=False),
            config.norm(out_channels),
        )

    def forward(self, imgs: Tensor):
        return imgs + self.main(imgs)


class ResNetGenerator(nn.Sequential):
    def __init__(self, config: CycleGANConfig):
        super().__init__()
        channels = config.base_channels

        self.input_block = nn.Sequential(
            conv7x7(config.in_channels, channels, bias=False),
            config.norm(channels),
            config.act(),
        )

        self.down_blocks = nn.Sequential()
        for _ in range(config.downsample):
            block = [
                conv3x3(channels, channels * 2, bias=False),
                config.norm(channels * 2),
                config.act(),
            ]
            self.down_blocks.extend(block)
            channels *= 2

        self.resnet_blocks = nn.Sequential()
        for _ in range(config.downsample):
            self.resnet_blocks.append(ResNetBlock(channels, channels, config))

        self.up_blocks = nn.Sequential()
        for _ in range(config.downsample):
            block = [
                upconv3x3(channels, channels // 2, bias=False),
                config.norm(channels // 2),
                config.act(),
            ]
            self.up_blocks.extend(block)
            channels //= 2

        self.output_block = nn.Sequential(
            conv7x7(channels, config.out_channels),
            nn.Tanh(),
        )
