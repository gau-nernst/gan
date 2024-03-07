# CycleGAN - https://arxiv.org/abs/1703.10593
#
# Code reference:
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

from torch import Tensor, nn

from .common import conv_norm_act
from .dcgan import init_weights


# NOTE: original code uses affine=False for InstanceNorm2d
class CycleGanResBlock(nn.Sequential):
    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__(
            conv_norm_act(dim, dim, 3, padding_mode="reflect", norm="instance", act="relu"),
            nn.Dropout(dropout),
            conv_norm_act(dim, dim, 3, padding_mode="reflect", norm="instance"),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + super().forward(x)


class ResNetGenerator(nn.Sequential):
    def __init__(self, base_dim: int = 64, n_blocks: int = 9, downsample: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.append(conv_norm_act(3, base_dim, 7, padding_mode="reflect", norm="instance", act="relu"))
        in_ch = base_dim

        for _ in range(downsample):
            self.append(conv_norm_act(in_ch, in_ch * 2, 3, 2), norm="instance", act="relu")
            in_ch *= 2

        self.extend([CycleGanResBlock(in_ch, dropout) for _ in range(n_blocks)])

        for _ in range(downsample):
            self.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, in_ch // 2, 3, 2, 1, 1, bias=False),
                    nn.InstanceNorm2d(in_ch // 2, affine=True),
                    nn.ReLU(inplace=True),
                )
            )
            in_ch //= 2

        self.append(conv_norm_act(base_dim, 3, 7, padding_mode="reflect", act="tanh"))  # no norm
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.apply(init_weights)
