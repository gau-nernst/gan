from functools import partial
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn


_Conv = Callable[..., nn.Module]
_Norm = Callable[[int], nn.Module]
_Act = Callable[[], nn.Module]

conv1x1 = partial(nn.Conv2d, kernel_size=1)
conv3x3 = partial(nn.Conv2d, kernel_size=3, padding=1)


def conv_norm_act(in_channels: int, out_channels: int, conv: _Conv, norm: _Norm, act: _Act, **kwargs):
    return nn.Sequential(
        conv(in_channels, out_channels, **kwargs, bias=False),
        norm(out_channels),
        act(),
    )
