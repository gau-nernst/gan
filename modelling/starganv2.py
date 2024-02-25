# https://arxiv.org/abs/1912.01865
# https://github.com/clovaai/stargan-v2
# NOTE: mask and HPF are not implemneted. See https://github.com/clovaai/stargan-v2/issues/70

import math

import torch
from torch import Tensor, nn

from .common import AdaIN, norm_act_conv


class StarGanv2EncoderBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, norm: str = "none", downsample: bool = False) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            *norm_act_conv(in_dim, in_dim, 3, norm=norm, act="leaky_relu"),
            nn.AvgPool2d(2) if downsample else nn.Identity(),
            *norm_act_conv(in_dim, out_dim, 3, norm=norm, act="leaky_relu"),
        )
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2) if downsample else nn.Identity(),
            nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.shortcut(x) + self.residual(x)


class StarGanv2DecoderBlock(nn.Module):
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


class StarGanv2Generator(nn.Module):
    def __init__(self, img_size: int = 256, style_dim: int = 64, base_dim: int = 64) -> None:
        super().__init__()
        self.in_conv = nn.Conv2d(3, base_dim, 3, 1, 1)
        in_dim = base_dim

        n_layers = int(math.log2(img_size)) - 4
        self.encoder = nn.Sequential()
        self.decoder = nn.ModuleList()

        for i in range(n_layers):
            out_dim = min(base_dim * 2 ** (i + 1), 512)
            self.encoder.append(StarGanv2EncoderBlock(in_dim, out_dim, norm="instance", downsample=True))
            in_dim = out_dim

        self.encoder.append(StarGanv2EncoderBlock(in_dim, in_dim, norm="instance"))
        self.encoder.append(StarGanv2EncoderBlock(in_dim, in_dim, norm="instance"))
        self.decoder.append(StarGanv2DecoderBlock(in_dim, in_dim, style_dim))
        self.decoder.append(StarGanv2DecoderBlock(in_dim, in_dim, style_dim))

        for i in range(n_layers - 1, -1, -1):
            out_dim = min(base_dim * 2**i, 512)
            self.decoder.append(StarGanv2DecoderBlock(in_dim, out_dim, style_dim, upsample=True))
            in_dim = out_dim

        assert in_dim == base_dim
        self.out_conv = norm_act_conv(base_dim, 3, 1, norm="instance", act="leaky_relu")

    def forward(self, x: Tensor, style: Tensor) -> Tensor:
        out = self.in_conv(x)
        out = self.encoder(out)
        for layer in self.decoder:
            out = layer(out, style)
        out = self.out_conv(out)
        return out


# Projection Discriminator
class StarGanv2Discriminator(nn.Sequential):
    def __init__(self, img_size: int = 256, n_domains: int = 2, base_dim: int = 64, output_dim: int = 1) -> None:
        super().__init__()
        self.append(nn.Conv2d(3, base_dim, 3, 1, 1))
        in_dim = base_dim

        n_layers = int(math.log2(img_size)) - 2
        for i in range(n_layers):
            out_dim = max(base_dim * 2 ** (i + 1), 512)
            self.append(StarGanv2EncoderBlock(in_dim, out_dim, downsample=True)),
            in_dim = out_dim

        self.append(
            nn.Sequential(
                nn.Flatten(),
                nn.LeakyReLU(0.2),
                nn.Linear(in_dim * 4 * 4, in_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(in_dim, n_domains * output_dim),
            )
        )

        self.n_domains = n_domains

    def forward(self, x: Tensor, domain: Tensor) -> Tensor:
        out = super().forward(x).unflatten(-1, (self.n_domains, -1))
        return out[torch.arange(x.shape[0], device=x.device), domain]


class MappingNetwork(nn.Module):
    def __init__(self, z_dim: int = 16, hidden_dim: int = 512, style_dim: int = 64, n_domains: int = 2) -> None:
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.ReLU())
        for _ in range(3):
            self.shared.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])

        self.unshared = nn.ModuleList()
        for _ in range(n_domains):
            branch = nn.Sequential()
            for _ in range(3):
                branch.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            branch.append(nn.Linear(hidden_dim, style_dim))
            self.unshared.append(branch)

    def forward(self, x: Tensor, domain: Tensor) -> Tensor:
        shared = self.shared(x)
        out = torch.stack([branch(shared) for branch in self.unshared], dim=1)
        return out[torch.arange(x.shape[0], device=x.device), domain]
