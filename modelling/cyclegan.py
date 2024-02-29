# CycleGAN - https://arxiv.org/abs/1703.10593
#
# Code reference:
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

from torch import Tensor, nn

from .common import conv_norm_act
from .pix2pix import init_weights


# NOTE: original code uses affine=False for InstanceNorm2d
class CycleGanResBlock(nn.Sequential):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__(
            *conv_norm_act(in_dim, out_dim, 3, norm="instance", act="relu"),
            *conv_norm_act(out_dim, out_dim, 3, norm="instance"),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + super().forward(x)


# TODO: use StarGAN instead?
class ResNetGenerator(nn.Sequential):
    def __init__(
        self, A_channels: int = 3, B_channels: int = 3, base_dim: int = 64, n_blocks: int = 9, downsample: int = 2
    ) -> None:
        super().__init__()
        self.append(conv_norm_act(A_channels, base_dim, 7, norm="instance", act="relu"))
        in_ch = base_dim

        for _ in range(downsample):
            self.append(conv_norm_act(in_ch, in_ch * 2, 3, 2), norm="instance", act="relu")
            in_ch *= 2

        for _ in range(n_blocks):
            self.append(CycleGanResBlock(in_ch))

        for _ in range(downsample):
            self.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, in_ch // 2, 3, 2, 1, 1, bias=False),
                    nn.InstanceNorm2d(in_ch // 2, affine=True),
                    nn.ReLU(inplace=True),
                )
            )
            in_ch //= 2

        self.append(conv_norm_act(base_dim, B_channels, 7, act="tanh"))  # no norm
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)
