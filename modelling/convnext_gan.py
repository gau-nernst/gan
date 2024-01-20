import math

import torch
from torch import Tensor, nn


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw_conv1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.pw_conv2 = nn.Linear(dim * 4, dim)
        self.gamma = nn.Parameter(torch.full((dim,), 1e-4))

    def forward(self, x: Tensor) -> Tensor:
        out = self.norm(self.dw_conv(x).permute(0, 2, 3, 1))
        out = self.pw_conv2(self.act(self.pw_conv1(out))) * self.gamma
        return x + out.permute(0, 3, 1, 2)


class ConvNeXtDiscriminator(nn.Sequential):
    def __init__(self, img_size: int, img_channels: int = 3, base_dim: int = 64) -> None:
        super().__init__()
        depth = int(math.log2(img_size // 8))
        in_ch = img_channels

        for i in range(depth):
            out_ch = base_dim if i == 0 else in_ch * 2
            self.append(nn.Conv2d(in_ch, out_ch, 2, 2))
            self.append(ConvNeXtBlock(out_ch))
            in_ch = out_ch

        self.append(nn.Conv2d(in_ch, 1, 8))
        self.append(nn.Flatten(0))

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)


class ConvNeXtGenerator(nn.Sequential):
    def __init__(self, img_size: int, img_channels: int = 3, z_dim: int = 128, base_dim: int = 64) -> None:
        depth = int(math.log2(img_size // 8))
        out_ch = base_dim * 2 ** (depth - 1)
        super().__init__(
            nn.Sequential(
                nn.Linear(z_dim, z_dim * 4),
                nn.GELU(),
                nn.Linear(z_dim * 4, z_dim),
                nn.GELU(),
                nn.Linear(z_dim, out_ch * 8 * 8),
                nn.Unflatten(-1, (-1, 8, 8)),
                nn.GELU(),
            ),
        )
        in_ch = out_ch

        for _ in range(depth):
            out_ch = in_ch // 2
            self.append(nn.ConvTranspose2d(in_ch, out_ch, 2, 2))
            self.append(ConvNeXtBlock(out_ch))
            in_ch = out_ch

        self.append(nn.Conv2d(in_ch, img_channels, 1))
        self.append(nn.Tanh())

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)


def init_weights(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)
