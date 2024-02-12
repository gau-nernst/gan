# SRGAN - https://arxiv.org/abs/1609.04802
#
# Original implementation uses Pixel Shuffle. It is equivalent to ConvTranspose
# https://arxiv.org/abs/1609.07009
# ConvTranspose is faster on CUDA most of the time. There is no reason to use Pixel Shuffle.

from torch import Tensor, nn

from .common import conv_norm_act
from .cyclegan import CycleGanResBlock


class SRResNet(nn.Module):
    def __init__(self, base_dim: int = 64, n_blocks: int = 16, n_upsample: int = 2) -> None:
        super().__init__()
        self.input_layer = conv_norm_act(3, base_dim, 9, act="prelu")

        self.blocks = nn.Sequential()
        for _ in range(n_blocks):
            self.blocks.append(CycleGanResBlock(base_dim))
        self.blocks.append(conv_norm_act(base_dim, base_dim, 3, norm="batch"))

        self.output_layer = nn.Sequential()
        for _ in range(n_upsample):
            self.output_layer.append(conv_norm_act(base_dim, base_dim, 6, 2, transpose=True, act="prelu"))
        self.output_layer.append(nn.Conv2d(base_dim, 3, 9, 1, 4))

    def forward(self, x: Tensor) -> Tensor:
        out = self.input_layer(x)
        out = out + self.blocks(out)
        out = self.output_layer(out)
        return out


class SRGANDiscriminator(nn.Sequential):
    def __init__(self, base_dim: int = 64):
        super().__init__()

        def get_nc(idx: int) -> int:
            return base_dim * 2 ** (idx // 2)

        self.append(conv_norm_act(3, base_dim, 3, act="leaky_relu"))

        for i in range(1, 8):
            self.append(conv_norm_act(get_nc(i - 1), get_nc(i), 3, i % 2 + 1, norm="batch", act="leaky_relu"))

        self.append(conv_norm_act(get_nc(7), get_nc(8), 1, act="leaky_relu"))
        self.append(nn.Conv2d(get_nc(8), 1, 1))
