from functools import partial
from typing import Callable, Optional

from torch import nn

_LEAKY_RELU_NEGATIVE_SLOPE = 0.2

def conv_norm_act(
    in_dim: int,
    out_dim: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    conv: Callable[..., nn.Module] = nn.Conv2d,
    norm: Optional[Callable[[int], nn.Module]] = nn.BatchNorm2d,
    act: Callable[[], nn.Module] = partial(nn.ReLU, inplace=True)
):
    return nn.Sequential(
        conv(in_dim, out_dim, kernel_size, stride, padding, bias=norm is None),
        norm(out_dim) if norm is not None else nn.Identity(),
        act(),
    )


def init_module(module: nn.Module, nonlinearity="relu"):
    for m in (module, *module.modules()):
        if isinstance(m, nn.modules.conv._ConvNd):
            nn.init.kaiming_normal_(m.weight, a=_LEAKY_RELU_NEGATIVE_SLOPE, nonlinearity=nonlinearity)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class LeakyReLU(nn.LeakyReLU):
    def __init__(self):
        super().__init__(_LEAKY_RELU_NEGATIVE_SLOPE, True)
