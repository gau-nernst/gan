# pix2pix - https://arxiv.org/abs/1611.07004
#
# Code reference:
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix


import torch
from torch import Tensor, nn

from .common import conv_norm_act
from .dcgan import init_weights


# NOTE: original code uses affine=False for InstanceNorm2d
class PatchGan(nn.Sequential):
    def __init__(self, base_dim: int = 64, depth: int = 3) -> None:
        super().__init__()
        self.append(conv_norm_act(6, base_dim, 4, 2, act="leaky_relu"))
        in_ch = base_dim

        for i in range(1, depth):
            out_ch = base_dim * 2 ** min(i, 3)
            self.append(conv_norm_act(in_ch, out_ch, 4, 2, norm="instance", act="leaky_relu"))
            in_ch = out_ch

        # original code uses kernel_size=4 here
        out_ch = base_dim * 2 ** min(depth, 3)
        self.append(conv_norm_act(in_ch, out_ch, 3, norm="instance", act="leaky_relu"))
        self.append(nn.Conv2d(out_ch, 1, 3, 1, 1))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.apply(init_weights)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return super().forward(torch.cat([x1, x2], dim=1))


class UnetGenerator(nn.Module):
    def __init__(self, depth: int = 7, base_dim: int = 64, dropout=0.0) -> None:
        super().__init__()
        self.down_blocks = nn.ModuleList()
        self.down_blocks.append(nn.Conv2d(3, base_dim, 4, 2, 1))
        in_ch = base_dim
        for i in range(1, depth - 1):
            out_ch = base_dim * 2 ** min(i, 3)
            block = nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_ch, out_ch, 4, 2, 1),
                nn.InstanceNorm2d(out_ch),
            )
            self.down_blocks.append(block)
            in_ch = out_ch

        out_ch = base_dim * 2 ** min(depth - 1, 3)
        self.inner_most = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_ch, out_ch, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_ch, in_ch, 4, 2, 1),
            nn.InstanceNorm2d(in_ch),
        )

        self.up_blocks = nn.ModuleList()
        for i in range(depth - 3, -1, -1):
            out_ch = base_dim * 2 ** min(i, 3)
            block = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_ch * 2, out_ch, 4, 2, 1),
                nn.InstanceNorm2d(out_ch),
                nn.Dropout(dropout),
            )
            self.up_blocks.append(block)
            in_ch = out_ch
        self.up_blocks.append(
            nn.Sequential(
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_ch * 2, 3, 4, 2, 1),
                nn.Tanh(),
            )
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        fmaps = [x]
        for down in self.down_blocks:
            fmaps.append(down(fmaps[-1]))

        out = self.inner_most(fmaps[-1])

        for up in self.up_blocks:
            out = up(torch.cat([out, fmaps.pop()], 1))
        return out
