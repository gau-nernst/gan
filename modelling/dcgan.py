# DCGAN - https://arxiv.org/abs/1701.07875
# Discriminator will downsample until the feature map is 4x4, then flatten and matmul
# Generator will matmul, reshape to 4x4, then upsample until the desired image size is obtained
#
# Code references:
# https://github.com/soumith/dcgan.torch/
# https://github.com/martinarjovsky/WassersteinGAN/

from functools import partial

from torch import Tensor, nn

from .base import _Act, _Norm, conv_norm_act


class Discriminator(nn.Module):
    def __init__(
        self,
        img_size: int = 64,
        img_channels: int = 3,
        init_map_size: int = 4,
        min_channels: int = 64,
        norm: _Norm = partial(nn.BatchNorm2d, track_running_stats=False),
        act: _Act = partial(nn.LeakyReLU, 0.2, True),
    ):
        super().__init__()
        self.layers = nn.Sequential()

        # add strided conv until image size = 4
        conv = partial(nn.Conv2d, kernel_size=4, stride=2, padding=1, bias=False)
        while img_size > init_map_size:
            self.layers.append(conv_norm_act(img_channels, min_channels, conv, norm, act))
            img_channels = min_channels
            min_channels *= 2
            img_size //= 2

        # flatten and matmul
        self.layers.append(nn.Conv2d(img_channels, 1, init_map_size))

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)

    def forward(self, imgs: Tensor):
        return self.layers(imgs).view(-1)


class Generator(nn.Module):
    def __init__(
        self,
        img_size: int = 64,
        img_channels: int = 3,
        z_dim: int = 128,
        init_map_size: int = 4,
        min_channels: int = 64,
        norm: _Norm = partial(nn.BatchNorm2d, track_running_stats=False),
        act: _Act = partial(nn.ReLU, True),
    ):
        super().__init__()
        self.layers = nn.Sequential()

        # matmul and reshape to 4x4
        channels = min_channels * img_size // 2 // init_map_size
        self.layers.append(conv_norm_act(z_dim, channels, nn.ConvTranspose2d, norm, act, kernel_size=init_map_size))

        # conv transpose until reaching image size / 2
        conv = partial(nn.ConvTranspose2d, kernel_size=4, stride=2, padding=1)
        while init_map_size < img_size // 2:
            self.layers.append(conv_norm_act(channels, channels // 2, conv, norm, act))
            channels //= 2
            init_map_size *= 2

        # last layer use tanh activation
        self.layers.extend([conv(channels, img_channels), nn.Tanh()])

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)

    def forward(self, z_embs: Tensor):
        return self.layers(z_embs[:, :, None, None])


def init_weights(module: nn.Module):
    if isinstance(module, nn.modules.conv._ConvNd):
        nn.init.normal_(module.weight, 0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.modules.batchnorm._BatchNorm):
        nn.init.normal_(module.weight, 1, 0.02)
        nn.init.zeros_(module.bias)
