import einops
import pytorch_lightning as pl
import torchvision.transforms as TT
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from modelling import init_module, make_layers
from training import GANSystem
from utils import _Act, _Norm, get_parser


class Discriminator(nn.Sequential):
    def __init__(self, norm: _Norm = nn.BatchNorm2d, act: _Act = nn.ReLU):
        super().__init__()
        kwargs = dict(kernel_size=3, padding=1, norm=norm, act=act)
        layer_configs = [[32], [64], [128], [256]]
        self.convs = nn.Sequential(*make_layers(1, layer_configs, **kwargs))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()


class Generator(nn.Module):
    def __init__(self, z_dim, norm: _Norm = nn.BatchNorm2d, act: _Act = nn.ReLU):
        super().__init__()
        self.z_dim = z_dim
        self.linear = nn.Linear(z_dim, 512 * 2 * 2)
        kwargs = dict(kernel_size=4, stride=2, padding=1, conv=nn.ConvTranspose2d, norm=norm, act=act)
        layer_configs = [[256], [128], [64], [32]]
        self.layers = nn.Sequential(
            *make_layers(512, layer_configs, **kwargs),
            nn.Conv2d(32, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = einops.rearrange(self.linear(x), "b (c h w) -> b c h w", h=2, w=2)
        return self.layers(x)


def main():
    parser = get_parser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    transform = TT.Compose([TT.Pad(2), TT.ToTensor(), TT.Normalize(0.5, 0.5)])
    ds = MNIST("data", transform=transform)
    dloader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=8)

    D = Discriminator()
    init_module(D, "relu")

    G = Generator(args.z_dim)
    init_module(G, "relu")

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
