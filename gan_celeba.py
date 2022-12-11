import datetime
from copy import deepcopy

import pytorch_lightning as pl
import torch
import torchvision.datasets as TD
import torchvision.transforms as TT
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.argparse import add_argparse_args
from torch import Tensor, nn
from torch.utils.data import DataLoader

from modelling import dcgan
from training import GANSystem, ImageLoggingCallback
from utils import get_parser


class RowNormalizer(nn.Module):
    def forward(self, x: Tensor):
        x = x.float()
        return x / x.mean(dim=1, keepdim=True)


def main():
    parser = get_parser()
    parser = add_argparse_args(GANSystem, parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    print("Arguments")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    img_size = args.img_size
    transform = TT.Compose([TT.Resize(img_size), TT.CenterCrop(img_size), TT.ToTensor(), TT.Normalize(0.5, 0.5)])
    ds = TD.CelebA("data", split="all", transform=transform)
    dloader = DataLoader(
        dataset=ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    celeba_n_feats = 40

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.logger = TensorBoardLogger("logs", name=args.log_name, version=timestamp)

    fixed_noise = torch.randn((40, args.z_dim))
    fixed_y = torch.randint(2, size=(40, celeba_n_feats))
    args.callbacks = [ImageLoggingCallback(1000, fixed_noise, fixed_y)]

    d_kwargs = dict(img_size=img_size, img_depth=3)
    if args.conditional:
        condition_dim = 32
        condition_encoder = nn.Sequential(
            RowNormalizer(),
            nn.Linear(celeba_n_feats, condition_dim),
        )
        nn.init.normal_(condition_encoder[1].weight, 0, 0.02)
        d_kwargs.update(c_dim=condition_dim, c_encoder=condition_encoder)
    g_kwargs = deepcopy(d_kwargs)

    # DCGAN
    D = dcgan.Discriminator(**d_kwargs)
    G = dcgan.Generator(z_dim=args.z_dim, **g_kwargs)
    print(D)
    print(G)

    gan = GANSystem(D, G, **vars(args))
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(gan, dloader)


if __name__ == "__main__":
    main()
