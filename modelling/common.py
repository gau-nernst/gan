from functools import partial

from torch import Tensor, nn


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        out = x.flatten(-2).transpose(-1, -2)
        out = super().forward(out)
        out = out.transpose(-1, -2).unflatten(-1, x.shape[-2:])
        return out


def get_norm(norm: str, dim: int):
    return dict(
        none=nn.Identity,
        batch=partial(nn.BatchNorm2d, track_running_stats=False),
        instance=partial(nn.InstanceNorm2d, affine=True),
        layer=LayerNorm2d,
    )[norm](dim)


def get_act(act: str):
    return dict(
        none=nn.Identity,
        relu=nn.ReLU,
        leaky_relu=partial(nn.LeakyReLU, negative_slope=0.2),
        prelu=nn.PReLU,
        tanh=nn.Tanh,
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
