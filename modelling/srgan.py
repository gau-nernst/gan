from functools import partial

from torch import Tensor, nn

from .base import _Act, _Norm, conv1x1, conv3x3, conv_norm_act, leaky_relu
from .cyclegan import ResNetBlock


class SRGANGenerator(nn.Module):
    def __init__(
        self,
        img_channels: int = 3,
        base_channels: int = 64,
        n_blocks: int = 16,
        upsample: int = 2,
        norm: _Norm = nn.BatchNorm2d,
        act: _Act = nn.PReLU,
    ):
        super().__init__()
        conv9x9 = partial(nn.Conv2d, kernel_size=9, padding=4)
        self.input_layer = nn.Sequential(conv9x9(img_channels, base_channels), act())

        self.blocks = nn.Sequential()
        for _ in range(n_blocks):
            self.blocks.append(ResNetBlock(base_channels, base_channels, norm, act))
        self.blocks.append(conv_norm_act(base_channels, base_channels, conv3x3, norm, nn.Identity))

        self.output_layer = nn.Sequential()
        for _ in range(upsample):
            self.output_layer.extend([conv3x3(base_channels, base_channels * 4), nn.PixelShuffle(2), act()])
        self.output_layer.append(conv9x9(base_channels, img_channels))

    def forward(self, imgs: Tensor):
        out = self.input_layer(imgs)
        out = out + self.blocks(out)
        out = self.output_layer(out)
        return out


class SRGANDiscriminator(nn.Sequential):
    def __init__(
        self,
        img_channels: int = 3,
        base_channels: int = 64,
        norm: _Norm = nn.BatchNorm2d,
        act: _Act = leaky_relu,
    ):
        super().__init__()

        def get_n_filters(idx: int):
            return base_channels * 2 ** (idx // 2)

        self.extend([conv3x3(img_channels, base_channels), act()])

        for i in range(1, 8):
            self.append(conv_norm_act(get_n_filters(i - 1), get_n_filters(i), conv3x3, norm, act, stride=i % 2 + 1))

        self.extend([conv1x1(get_n_filters(7), get_n_filters(8)), act(), conv1x1(get_n_filters(8), 1)])
