from typing import Callable, Optional

from torch import nn

_Norm = Callable[[int], nn.Module]
_Act = Callable[[], nn.Module]


def conv_norm_act(
    in_dim: int,
    out_dim: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    conv: Callable[..., nn.Module] = nn.Conv2d,
    norm: Optional[_Norm] = nn.BatchNorm2d,
    act: Optional[_Act] = nn.ReLU,
):
    return nn.Sequential(
        conv(in_dim, out_dim, kernel_size, stride, padding, bias=norm is None),
        norm(out_dim) if norm is not None else nn.Identity(),
        act() if act is not None else nn.Identity(),
    )


def make_layers(in_dim: int, layer_configs, **kwargs):
    layers = []
    for out_dim, *args in layer_configs:
        layers.append(conv_norm_act(in_dim, out_dim, *args, **kwargs))
        in_dim = out_dim
    return layers
