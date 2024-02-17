# https://arxiv.org/abs/1912.01865
# https://github.com/clovaai/stargan-v2

import math

from torch import Tensor, nn

from .common import AdaIN, norm_act_conv


class StarGanv2ResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, norm: str = "none", downsample: bool = False) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            norm_act_conv(in_dim, in_dim, 3, norm=norm, act="leaky_relu"),
            nn.AvgPool2d(2) if downsample else nn.Identity(),
            norm_act_conv(in_dim, out_dim, 3, norm=norm, act="leaky_relu"),
        )
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2) if downsample else nn.Identity(),
            nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.shortcut(x) + self.residual(x)


class StarGanv2UpsampleResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, style_dim: int = 64, upsample: bool = False) -> None:
        super().__init__()
        self.residual = nn.ModuleList(
            [
                AdaIN(in_dim, style_dim),
                nn.LeakyReLU(0.2),
                nn.Conv2d(in_dim, out_dim, 3, 1, 1),
                nn.Upsample(scale_factor=2.0) if upsample else nn.Identity(),
                AdaIN(out_dim, style_dim),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_dim, out_dim, 3, 1, 1),
            ]
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity(),
            nn.Upsample(scale_factor=2.0) if upsample else nn.Identity(),
        )

    def forward(self, x: Tensor, style: Tensor) -> Tensor:
        res = x
        for layer in self.residual:
            res = layer(res, style) if isinstance(layer, AdaIN) else layer(res)
        return self.shortcut(x) + res


class StarGanv2Generator(nn.Sequential):
    def __init__(self, img_size: int = 256, style_dim: int = 64) -> None:
        self.in_conv = nn.Conv2d(3, 64, 3, 1, 1)
        in_dim = 64

        self.encoder = nn.Sequential()

        n_layers = int(math.log2(img_size)) - 4
        for _ in range(n_layers):
            out_dim = min(in_dim * 2, 512)
            self.encoder.append(StarGanv2ResBlock(in_dim, out_dim, norm="instance", downsample=True))
            in_dim = out_dim
        self.encoder.append(StarGanv2ResBlock(in_dim, in_dim, norm="instance"))

        self.decoder = nn.Sequential()
        self.decoder.append(StarGanv2UpsampleResBlock)

        self.out_conv = norm_act_conv(64, 3, 1, norm="instance", act="leaky_relu")
