from functools import partial
from typing import Callable, List, Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_Conv = Callable[..., nn.modules.conv._ConvNd]
_Norm = Callable[[int], nn.Module]
_Act = Callable[[], nn.Module]


def conv_norm_act(
    in_dim: int,
    out_dim: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    bias: bool = True,
    order: Optional[List[Literal["conv", "norm", "act"]]] = None,
    conv: _Conv = nn.Conv2d,
    norm: Optional[_Norm] = nn.BatchNorm2d,
    act: Optional[_Act] = partial(nn.ReLU, inplace=True),
):
    if order is None:
        order = ["conv", "norm", "act"]
    layers = nn.Sequential()
    mapping = dict(
        conv=partial(conv, in_dim, out_dim, kernel_size, stride, padding, bias=bias),
        norm=partial(norm, in_dim),
        act=act,
    )
    for name in order:
        layers.append(mapping[name]())
        if name == "conv":
            mapping.update(norm=partial(norm, out_dim))
    return layers


class PixelNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x)


class Blur(nn.Module):
    def __init__(self, kernel: Optional[List[float]] = None):
        super().__init__()
        kernel = torch.tensor(kernel or [1, 2, 1], dtype=torch.float)
        kernel = kernel.view(1, -1) * kernel.view(-1, 1)
        kernel = kernel[None, None] / kernel.sum()
        self.register_buffer("kernel", kernel)

    def forward(self, imgs: Tensor):
        channels = imgs.shape[1]
        kernel = self.kernel.expand(channels, -1, -1, -1)
        return F.conv2d(imgs, kernel, padding="same", groups=channels)
