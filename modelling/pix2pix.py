from functools import partial

from torch import nn

from .base import _Act, _Norm, conv_norm_act, leaky_relu


class PatchGAN(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        n_layers: int = 3,
        norm: _Norm = nn.InstanceNorm2d,
        act: _Act = leaky_relu,
    ):
        super().__init__()
        conv = partial(nn.Conv2d, kernel_size=4, padding=1)

        def get_out_c(idx):
            return base_channels * 2 ** min(idx, 3)

        self.append(nn.Sequential(conv(in_channels, base_channels, stride=2), act()))

        for i in range(1, n_layers):
            self.append(conv_norm_act(get_out_c(i - 1), get_out_c(i), conv, norm, act, stride=2))

        self.append(conv_norm_act(get_out_c(n_layers - 1), get_out_c(n_layers), conv, norm, act))
        self.append(conv(get_out_c(n_layers), 1))
