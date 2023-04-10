from functools import partial
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
from torch import nn


_Conv = Callable[..., nn.Module]
_Norm = Callable[[int], nn.Module]
_Act = Callable[[], nn.Module]

conv3x3 = partial(nn.Conv2d, kernel_size=3, padding=1)
conv1x1 = partial(nn.Conv2d, kernel_size=1)

batched_conv2d = torch.vmap(F.conv2d)


def conv_norm_act(
    in_dim: int,
    out_dim: int,
    order: Optional[List[str]] = None,
    conv: _Conv = conv3x3,
    norm: _Norm = partial(nn.BatchNorm2d, track_running_stats=False),
    act: _Act = partial(nn.ReLU, inplace=True),
):
    if order is None:
        order = ["conv", "norm", "act"]
    layers = nn.Sequential()
    mapping = dict(
        conv=partial(conv, in_dim, out_dim),
        norm=partial(norm, in_dim),
        act=act,
    )
    for name in order:
        assert name in ("conv", "norm", "act")
        layers.append(mapping[name]())
        if name == "conv":
            mapping.update(norm=partial(norm, out_dim))
    return layers
