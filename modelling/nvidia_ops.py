import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.cuda.amp.autocast_mode import custom_bwd, custom_fwd

from .base import conv1x1


# introduced in Progressive GAN. only used in Progressive GAN Generator
class PixelNorm(nn.Module):
    def __init__(self, in_dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = in_dim**0.5
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, eps=self.eps) * self.scale


# introduced in Progressive GAN
class MinibatchStdDev(nn.Module):
    def __init__(self, group_size: int = 4):
        super().__init__()
        self.group_size = group_size

    def forward(self, imgs: Tensor):
        _, c, h, w = imgs.shape
        std = imgs.view(self.group_size, -1, c, h, w).std(dim=0, unbiased=False)
        std = std.mean([1, 2, 3], keepdim=True).repeat(self.group_size, 1, h, w)
        return torch.cat([imgs, std], dim=1)


# introduced in Progressive GAN
class EqualizedLR(nn.Module):
    def __init__(self, weight: Tensor):
        super().__init__()
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        gain = 2**0.5  # use gain=sqrt(2) everywhere
        self.scale = gain / fan_in**0.5

    def forward(self, weight: Tensor):
        return weight * self.scale

    def extra_repr(self) -> str:
        return f"scale={self.scale}"


# introduced in StyleGAN
def upfirdn2d(imgs: Tensor, kernel: Tensor, up: int, down: int, px1: int, px2: int, py1: int, py2: int):
    n, c, h, w = imgs.shape
    ky, kx = kernel.shape
    kernel = kernel.view(1, 1, ky, kx).expand(c, 1, ky, kx)

    if up > 1:
        _imgs = imgs.new_zeros(n, c, h * up, w * up)
        _imgs[:, :, ::up, ::up] = imgs
        imgs = _imgs

    if px1 == px2 and py1 == py2:
        return F.conv2d(imgs, kernel, stride=down, padding=(py1, px1), groups=c)
    else:
        return F.conv2d(F.pad(imgs, (px1, px2, py1, py2)), kernel, stride=down, groups=c)


# significant speed-up for higher order gradients
# https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html
class UpFIRDn2d(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, imgs, kernel, up, down, px1, px2, py1, py2):
        ctx.save_for_backward(kernel.flip(0, 1))
        ctx.others = (up, down, px1, px2, py1, py2)
        return upfirdn2d(imgs, kernel, up, down, px1, px2, py1, py2)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        (kernel,) = ctx.saved_tensors
        up, down, px1, px2, py1, py2 = ctx.others
        grad_imgs = UpFIRDn2d.apply(grad_output, kernel, down, up, px2, px1, py2, py1)
        return grad_imgs, *[None] * 7


class Blur(nn.Module):
    def __init__(self, kernel_size: int = 3, up: int = 1, down: int = 1):
        super().__init__()
        self.up = up
        self.down = down
        self.p1 = (kernel_size - 1) // 2
        self.p2 = kernel_size - 1 - self.p1
        self.register_buffer("kernel", self.make_kernel(kernel_size))
        self.kernel: Tensor

    def forward(self, imgs: Tensor):
        # gradfix version is only faster for 2nd and higher order gradient
        fn = UpFIRDn2d.apply if self.training else upfirdn2d
        return fn(imgs, self.kernel, self.up, self.down, self.p1, self.p2, self.p1, self.p2)

    @staticmethod
    def make_kernel(kernel_size: int):
        # https://github.com/adobe/antialiased-cnns/blob/master/antialiased_cnns/blurpool.py
        # example kernels: [1,2,1], [1,3,3,1]
        assert 2 <= kernel_size <= 7
        kernel = [math.comb(kernel_size - 1, i) for i in range(kernel_size)]
        kernel = torch.tensor(kernel, dtype=torch.float) / sum(kernel)
        return kernel.view(1, -1) * kernel.view(-1, 1)


def up_conv_blur(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    blur_size: Optional[int] = 3,
):
    padding = (kernel_size - 1) // 2
    output_padding = kernel_size % 2
    if blur_size is None:  # progressive GAN
        layers = nn.Sequential(
            nn.Upsample(scale_factor=2.0),
            nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding),
        )
    elif kernel_size == 1:  # this is never used
        layers = nn.Sequential(conv1x1(in_channels, out_channels), Blur(blur_size, up=2))
    else:
        layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 2, padding, output_padding),
            Blur(blur_size),
        )
    return layers


def blur_conv_down(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    blur_size: Optional[int] = 3,
):
    padding = (kernel_size - 1) // 2
    if blur_size is None:  # progressive GAN
        layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding),
            nn.AvgPool2d(2),
        )
    elif kernel_size == 1:
        layers = nn.Sequential(Blur(blur_size, down=2), conv1x1(in_channels, out_channels))
    else:
        layers = nn.Sequential(
            Blur(blur_size),
            nn.Conv2d(in_channels, out_channels, kernel_size, 2, padding),
        )
    return layers
