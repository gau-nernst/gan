# CycleGAN - https://arxiv.org/abs/1703.10593
#
# Code reference:
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

from torch import Tensor, nn

from .pix2pix import init_weights


# almost identical to torchvision.models.resnet.BasicBlock
class ResNetBlock(nn.Sequential):
    def __init__(self, dim: int) -> None:
        super().__init__(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x: Tensor):
        return x + super().forward(x)


class ResNetGenerator(nn.Sequential):
    def __init__(
        self, A_channels: int = 3, B_channels: int = 3, base_dim: int = 64, n_blocks: int = 9, downsample: int = 2
    ) -> None:
        super().__init__()
        self.append(nn.Conv2d(A_channels, base_dim, 7, 1, 3))
        self.append(nn.InstanceNorm2d(base_dim))
        self.append(nn.ReLU(inplace=True))
        in_ch = base_dim

        for _ in range(downsample):
            self.append(nn.Conv2d(in_ch, in_ch * 2, 3, 2, 1))
            self.append(nn.InstanceNorm2d(in_ch * 2))
            self.append(nn.ReLU(inplace=True))
            in_ch *= 2

        for _ in range(n_blocks):
            self.append(ResNetBlock(in_ch))

        for _ in range(downsample):
            self.append(nn.ConvTranspose2d(in_ch, in_ch // 2, 3, 2, 1, 1))
            self.append(nn.InstanceNorm2d(in_ch // 2))
            self.append(nn.ReLU(inplace=True))
            in_ch //= 2

        self.append(nn.Conv2d(base_dim, B_channels, 7, 1, 3))
        self.append(nn.Tanh())

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)
