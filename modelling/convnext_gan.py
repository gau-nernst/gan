import torch
from torch import nn, Tensor
import math


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


class ConvNeXtDiscriminator(nn.Module):
    def __init__(self, img_size: int, img_channels: int = 3, n_classes: int = 1, base_dim: int = 64) -> None:
        super().__init__()
        depth = int(math.log2(img_size // 8))

        self.blocks = nn.Sequential()
        in_ch = img_channels

        for i in range(depth):
            out_ch = base_dim if i == 0 else in_ch * 2
            self.blocks.append(ConvNeXtBlock(in_ch))
            in_ch = out_ch

        self.blocks.append(
            nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, in_ch, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, in_ch, 3, 1, 1),
                nn.ReLU(inplace=True),
            )
        )

    def forward(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        if y is None:
            y = x.new_zeros(1, dtype=torch.long)
        embs = self.blocks(x).sum(dim=(-1, -2))
        return (embs * self.y_embs(y)).sum(1)
