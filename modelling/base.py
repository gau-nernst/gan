from functools import partial
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_Conv = Callable[..., nn.modules.conv._ConvNd]
_Norm = Callable[[int], nn.Module]
_Act = Callable[[], nn.Module]


def conv_act(
    in_dim: int,
    out_dim: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    conv: _Conv = nn.Conv2d,
    act: _Act = partial(nn.ReLU, inplace=True),
):
    return nn.Sequential(conv(in_dim, out_dim, kernel_size, stride, padding), act())


def conv_norm_act(
    in_dim: int,
    out_dim: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    conv: _Conv = nn.Conv2d,
    norm: Optional[_Norm] = nn.BatchNorm2d,
    act: Optional[_Act] = partial(nn.ReLU, inplace=True),
):
    return nn.Sequential(
        conv(in_dim, out_dim, kernel_size, stride, padding, bias=norm is None),
        norm(out_dim) if norm is not None else nn.Identity(),
        act() if act is not None else nn.Identity(),
    )


def conv_act_norm(
    in_dim: int,
    out_dim: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    conv: _Conv = nn.Conv2d,
    norm: Optional[_Norm] = nn.BatchNorm2d,
    act: Optional[_Act] = partial(nn.ReLU, inplace=True),
):
    return nn.Sequential(
        conv(in_dim, out_dim, kernel_size, stride, padding),
        act() if act is not None else nn.Identity(),
        norm(out_dim) if norm is not None else nn.Identity(),
    )


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
