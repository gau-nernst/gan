from functools import partial

from torch import nn


def get_norm(norm: str, dim: int):
    return dict(
        none=nn.Identity,
        batch=partial(nn.BatchNorm2d, track_running_stats=False),
        instance=partial(nn.InstanceNorm2d, affine=True),
    )[norm](dim)


def get_act(act: str):
    return dict(
        relu=nn.ReLU,
        leaky_relu=partial(nn.LeakyReLU, negative_slope=0.2),
    )[act]()


def conv_norm_act(
    in_dim: int,
    out_dim: int,
    kernel_size: int,
    stride: int = 1,
    transpose: bool = False,
    norm: str = "none",
    act: str = "none",
) -> nn.Sequential:
    conv_cls = nn.ConvTranspose2d if transpose else nn.Conv2d
    return nn.Sequential(
        conv_cls(in_dim, out_dim, kernel_size, stride, (kernel_size - 1) // 2, bias=norm in ("none",)),
        get_norm(norm, out_dim),
        get_act(act),
    )
