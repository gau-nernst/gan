# https://arxiv.org/abs/1711.09020
# https://github.com/yunjey/stargan

from torch import Tensor, nn


def _conv_norm_act(in_dim: int, out_dim: int, kernel_size: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, (kernel_size - 1) // 2, bias=False),
        nn.InstanceNorm2d(out_dim, affine=True, track_running_stats=True),
        nn.ReLU(),
    )


class StarGanResBlock(nn.Sequential):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__(
            *_conv_norm_act(in_dim, out_dim, 3),
            nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_dim, affine=True, track_running_stats=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + super().forward(x)


def _upsample(in_dim: int, out_dim: int):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1, bias=False),
        nn.InstanceNorm2d(out_dim, affine=True, track_running_stats=True),
        nn.ReLU(),
    )


class StarGanGenerator(nn.Sequential):
    def __init__(self, n_classes: int = 0, n_res_layers: int = 6) -> None:
        super().__init__(
            _conv_norm_act(3 + n_classes, 64, 7),
            _conv_norm_act(64, 128, 4, 2),
            _conv_norm_act(128, 256, 4, 2),
            *[StarGanResBlock(256, 256) for _ in range(n_res_layers)],
            _upsample(256, 128),
            _upsample(128, 64),
            nn.Sequential(nn.Conv2d(64, 3, 7, 1, 3), nn.Tanh()),
        )


class StarGanDiscriminator(nn.Module):
    def __init__(self, img_size: int, n_classes: int, n_layers: int = 6) -> None:
        super().__init__()
        self.backbone = nn.Sequential()
        in_dim = 3
        out_dim = 64
        for _ in range(n_layers):
            self.backbone.extend([nn.Conv2d(in_dim, out_dim, 4, 2, 1), nn.LeakyReLU()])
            in_dim = out_dim
            out_dim *= 2

        self.out1 = nn.Conv2d(in_dim, 1, 3, 1, 1)
        self.out2 = nn.Conv2d(in_dim, n_classes, img_size // 2**n_layers)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        out = self.backbone(x)
        return self.out1(out), self.out2(out).flatten(1)
