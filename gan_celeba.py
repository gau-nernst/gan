import einops
import pytorch_lightning as pl
import torchvision.datasets as TD
import torchvision.transforms as TT
from torch import nn
from torch.utils.data import DataLoader

from modelling import LeakyReLU, init_module, make_layers
from training import GANSystem
from utils import _Act, _Norm, get_parser


class Discriminator(nn.Sequential):
    def __init__(self, norm: _Norm = nn.BatchNorm2d, act: _Act = nn.ReLU):
        super().__init__()
        kwargs = dict(kernel_size=3, stride=2, padding=1, norm=norm, act=act)
        layer_configs = [[64], [128], [256], [512]]
        self.convs = nn.Sequential(*make_layers(3, layer_configs, **kwargs))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()


class Generator(nn.Sequential):
    def __init__(self, z_dim: int, norm: _Norm = nn.BatchNorm2d, act: _Act = nn.ReLU):
        super().__init__()
        self.z_dim = z_dim
        kwargs = dict(kernel_size=4, stride=2, padding=1, conv=nn.ConvTranspose2d, norm=norm, act=act)
        layer_configs = [[1024], [512], [256], [128], [64]]
        self.ups = nn.Sequential(*make_layers(z_dim // 4, layer_configs, **kwargs))
        self.out_conv = nn.Conv2d(64, 3, 3, stride=1, padding=1)

    def forward(self, x):
        x = einops.rearrange(x, "b (c h w) -> b c h w", h=2, w=2)
        return super().forward(x)


def main():
    parser = get_parser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    transform = TT.Compose([TT.Resize(64), TT.CenterCrop(64), TT.ToTensor(), TT.Normalize(0.5, 0.5)])
    ds = TD.CelebA("data", split="all", transform=transform)
    dloader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    D = Discriminator(act=LeakyReLU)
    init_module(D, "leaky_relu")

    G = Generator(args.z_dim, act=LeakyReLU)
    init_module(G, "leaky_relu")

    gan = GANSystem(
        discriminator=D,
        generator=G,
        z_dim=args.z_dim,
        method=args.method,
        log_img_interval=args.log_interval,
    )
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(gan, dloader)


if __name__ == "__main__":
    main()
