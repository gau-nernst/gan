from functools import partial

from torch import nn


def get_norm(dim: int, norm: str) -> nn.BatchNorm2d | nn.Identity:
    norm_cls = dict(
        bn=partial(nn.BatchNorm2d, track_running_stats=False),
        none=nn.Identity,
    )[norm]
    return norm_cls(dim)
