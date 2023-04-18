# Code reference:
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

from functools import partial

import torch
from torch import Tensor, nn

from .base import _Act, _Norm, conv3x3, conv_norm_act
from .dcgan import init_weights
from .pix2pix import PatchGAN


# almost identical to torchvision.models.resnet.BasicBlock
class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float, norm: _Norm, act: _Act):
        super().__init__()
        self.main = nn.Sequential(
            conv_norm_act(in_channels, out_channels, conv3x3, norm, act),
            nn.Dropout(dropout),
            conv_norm_act(out_channels, out_channels, conv3x3, norm, nn.Identity),
        )

    def forward(self, imgs: Tensor):
        return imgs + self.main(imgs)


class ResNetGenerator(nn.Sequential):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        n_blocks: int = 9,
        downsample: int = 2,
        dropout: float = 0.5,
        norm: _Norm = nn.InstanceNorm2d,
        act: _Act = partial(nn.ReLU, inplace=True),
    ):
        super().__init__()
        conv7x7 = partial(nn.Conv2d, kernel_size=7, padding=3)
        upconv3x3 = partial(nn.ConvTranspose2d, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.input_block = conv_norm_act(in_channels, base_channels, conv7x7, norm, act)

        self.down_blocks = nn.Sequential()
        for _ in range(downsample):
            self.down_blocks.append(conv_norm_act(base_channels, base_channels * 2, conv3x3, norm, act))
            base_channels *= 2

        self.resnet_blocks = nn.Sequential()
        for _ in range(n_blocks):
            self.resnet_blocks.append(ResNetBlock(base_channels, base_channels, dropout, norm, act))

        self.up_blocks = nn.Sequential()
        for _ in range(downsample):
            self.up_blocks.append(conv_norm_act(base_channels, base_channels // 2, upconv3x3, norm, act))
            base_channels //= 2

        self.output_block = nn.Sequential(conv7x7(base_channels, out_channels), nn.Tanh())

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)


class CycleGANGenerator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = ResNetGenerator(**kwargs)

    def forward(self, z_embs: Tensor, ys: Tensor):
        return self.model(ys)


class CycleGANDiscriminator(nn.Module):
    def __init__(self, img_channels: int, y_channels: int, **kwargs):
        super().__init__()
        self.model = PatchGAN(img_channels + y_channels, **kwargs)

    def forward(self, imgs: Tensor, ys: Tensor):
        return self.model(torch.cat([imgs, ys], dim=1))
