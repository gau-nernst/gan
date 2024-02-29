# https://arxiv.org/abs/1711.09020
# https://github.com/yunjey/stargan

from torch import Tensor, nn

from .common import conv_norm_act


class StarGanResBlock(nn.Sequential):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__(
            *conv_norm_act(in_dim, out_dim, 3, norm="instance", act="relu"),
            *conv_norm_act(out_dim, out_dim, 3, norm="instance"),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + super().forward(x)


class StarGanGenerator(nn.Sequential):
    def __init__(self, n_classes: int = 0, n_res_layers: int = 6) -> None:
        super().__init__(
            conv_norm_act(3 + n_classes, 64, 7, norm="instance", act="relu"),
            conv_norm_act(64, 128, 4, 2, norm="instance", act="relu"),
            conv_norm_act(128, 256, 4, 2, norm="instance", act="relu"),
            *[StarGanResBlock(256, 256) for _ in range(n_res_layers)],
            conv_norm_act(256, 128, 4, 2, transpose=True, norm="instance", act="relu"),
            conv_norm_act(128, 64, 4, 2, transpose=True, norm="instance", act="relu"),
            conv_norm_act(64, 3, 7, act="tanh"),  # no norm
        )


# NOTE: original paper uses negative_slop=0.01 (PyTorch's default) for LeakyReLU
class StarGanDiscriminator(nn.Module):
    def __init__(self, img_size: int, n_classes: int, n_layers: int = 6) -> None:
        super().__init__()
        self.backbone = nn.Sequential()
        in_dim = 3
        out_dim = 64
        for _ in range(n_layers):
            self.backbone.append(conv_norm_act(in_dim, out_dim, 4, 2, act="leaky_relu"))  # no norm
            in_dim = out_dim
            out_dim *= 2

        self.out1 = nn.Conv2d(in_dim, 1, 3, 1, 1)
        self.out2 = nn.Conv2d(in_dim, n_classes, img_size // 2**n_layers)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        out = self.backbone(x)
        return self.out1(out), self.out2(out).flatten(1)
