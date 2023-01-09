# Progressive GAN - https://arxiv.org/pdf/1710.10196
# See Table 2 for detailed model architecture
# Discriminator supports BlurConv (StyleGAN) and skip-connections (StyleGAN2)
# Generator has an option to use BlurConv
# Not implemented features
# - Progressive growing
#
# Code reference:
# https://github.com/tkarras/progressive_growing_of_gans

import math
from functools import partial
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import parametrize

from .base import _Act, _Norm, conv1x1, conv3x3, conv_norm_act


# implement FIR filter as depth-wise convolution
# reference: https://github.com/NVlabs/stylegan/blob/master/training/networks_stylegan.py#L22
def _upfirdn2d(imgs: Tensor, kernel: Tensor, up: int, down: int, padding: int):
    n, c, h, w = imgs.shape
    ky, kx = kernel.shape
    kernel = kernel.view(1, 1, ky, kx).expand(c, 1, ky, kx)
    if up > 1:
        _imgs = imgs.new_zeros(n, c, h * up, w * up)
        _imgs[:, :, ::up, ::up] = imgs
        imgs = _imgs
    return F.conv2d(imgs, kernel, stride=down, padding=padding, groups=c)


# backward pass for FIR filter is identical to its forward pass
# override backward() to significantly speed up backward pass
# gradient wrt conv kernel doesn't need to be computed
# reference: https://github.com/NVlabs/stylegan/blob/master/training/networks_stylegan.py#L96
class UpFIRDn2d(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, imgs: Tensor, kernel: Tensor, up: int, down: int, padding: int):
        ctx.save_for_backward(kernel.flip(0, 1))
        ctx.up = up
        ctx.down = down
        ctx.padding = padding
        return _upfirdn2d(imgs, kernel, up, down, padding)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_outputs: Tensor):
        (kernel,) = ctx.saved_tensors
        grad_inputs = _upfirdn2d(grad_outputs, kernel, ctx.down, ctx.up, ctx.padding)
        print(grad_outputs.shape, grad_inputs.shape)
        return grad_inputs, None, None, None, None


class Blur(nn.Module):
    def __init__(self, kernel_size: int = 3, up: int = 1, down: int = 1):
        super().__init__()
        self.up = up
        self.down = down
        self.padding = (kernel_size - 1) // 2
        self.register_buffer("kernel", self.make_kernel(kernel_size))

    def forward(self, imgs: Tensor):
        return UpFIRDn2d.apply(imgs, self.kernel, self.up, self.down, self.padding)

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
    if blur_size is None:
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
    if blur_size is None:
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


class PixelNorm(nn.Module):
    def __init__(self, in_dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = in_dim**0.5
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, eps=self.eps) * self.scale


class MinibatchStdDev(nn.Module):
    def __init__(self, group_size: int = 4):
        super().__init__()
        self.group_size = group_size

    def forward(self, imgs: Tensor):
        _, c, h, w = imgs.shape
        std = imgs.view(self.group_size, -1, c, h, w).std(dim=0, unbiased=False)
        std = std.mean([1, 2, 3], keepdim=True).repeat(self.group_size, 1, h, w)
        return torch.cat([imgs, std], dim=1)


class DiscriminatorStage(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        residual: bool = False,
        act: _Act = partial(nn.LeakyReLU, 0.2, True),
        blur_size: Optional[int] = None,
    ):
        super().__init__()
        conv_act = partial(conv_norm_act, order=["conv", "act"], conv=conv3x3, act=act)
        down_conv = partial(blur_conv_down, blur_size=blur_size)

        self.main = nn.Sequential(
            conv_act(in_dim, in_dim),
            conv_act(in_dim, out_dim, conv=partial(down_conv, kernel_size=3)),
        )
        self.shortcut = down_conv(in_dim, out_dim, 1) if residual else None  # skip-connection in StyleGAN2

    def forward(self, imgs: Tensor):
        out = self.main(imgs)
        if self.shortcut is not None:  # skip-connection in StyleGAN2
            out = (out + self.shortcut(imgs)) * 2 ** (-0.5)
        return out


class Discriminator(nn.Module):
    def __init__(
        self,
        img_size,
        img_depth: int = 3,
        base_depth: int = 16,
        max_depth: int = 512,
        smallest_map_size: int = 4,
        residual: bool = False,
        act: _Act = partial(nn.LeakyReLU, 0.2, True),
        blur_size: Optional[int] = None,
    ):
        assert img_size > 4 and math.log2(img_size).is_integer()
        super().__init__()
        conv_act = partial(conv_norm_act, order=["conv", "act"], conv=conv3x3, act=act)
        stage = partial(DiscriminatorStage, act=act, residual=residual, blur_size=blur_size)

        self.layers = nn.Sequential()
        self.layers.append(conv_act(img_depth, base_depth, conv=conv1x1))

        while img_size > smallest_map_size:
            out_depth = min(base_depth * 2, max_depth)
            self.layers.append(stage(base_depth, out_depth))
            base_depth = out_depth
            img_size //= 2

        self.layers.append(MinibatchStdDev())
        self.layers.append(conv_act(base_depth + 1, base_depth))
        self.layers.append(conv_act(base_depth, base_depth, conv=partial(nn.Conv2d, kernel_size=smallest_map_size)))
        self.layers.append(conv1x1(base_depth, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)

    def forward(self, imgs: Tensor):
        return self.layers(imgs).view(-1)


class Generator(nn.Module):
    def __init__(
        self,
        img_size: int,
        img_depth: int = 3,
        z_dim: int = 512,
        base_depth: int = 16,
        max_depth: int = 512,
        smallest_map_size: int = 4,
        norm: _Norm = PixelNorm,
        act: _Act = partial(nn.LeakyReLU, 0.2, True),
        blur_size: Optional[int] = None,
    ):
        assert img_size > 4 and math.log2(img_size).is_integer()
        super().__init__()
        up_conv3x3 = partial(up_conv_blur, kernel_size=3, blur_size=blur_size)
        conv_act_norm = partial(conv_norm_act, order=["conv", "act", "norm"], conv=conv3x3, act=act, norm=norm)

        in_depth = z_dim
        depth = base_depth * img_size // smallest_map_size
        out_depth = min(depth, max_depth)

        self.layers = nn.Sequential()
        self.layers.append(norm(in_depth))
        self.layers.append(
            conv_act_norm(in_depth, out_depth, conv=partial(nn.ConvTranspose2d, kernel_size=smallest_map_size))
        )
        self.layers.append(conv_act_norm(out_depth, out_depth))
        in_depth = out_depth
        depth //= 2

        while smallest_map_size < img_size:
            out_depth = min(depth, max_depth)
            self.layers.append(conv_act_norm(in_depth, out_depth, conv=up_conv3x3))
            self.layers.append(conv_act_norm(out_depth, out_depth))
            in_depth = out_depth
            depth //= 2
            smallest_map_size *= 2

        self.layers.append(conv1x1(in_depth, img_depth))

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)

    def forward(self, z_embs: Tensor):
        return self.layers(z_embs[:, :, None, None])


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


def init_weights(module: nn.Module):
    if isinstance(module, (nn.modules.conv._ConvNd, nn.Linear)):
        nn.init.normal_(module.weight)
        parametrize.register_parametrization(module, "weight", EqualizedLR(module.weight))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
