import datetime
from copy import deepcopy

import pytorch_lightning as pl
import torch
import torchvision.datasets as TD
import torchvision.transforms as TT
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.argparse import add_argparse_args
from torch import nn
from torch.utils.data import DataLoader

from modelling import dcgan
from training import GANSystem, ImageLoggingCallback
from utils import get_parser


def main():
    parser = get_parser()
    parser = add_argparse_args(GANSystem, parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    print("Arguments")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    # pad from 28x28 to 32x32
    transform = TT.Compose([TT.Pad(2), TT.ToTensor(), TT.Normalize(0.5, 0.5)])
    ds = TD.MNIST("data", transform=transform)
    dloader = DataLoader(
        dataset=ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.logger = TensorBoardLogger("logs", name=args.log_name, version=timestamp)

    fixed_noise = torch.randn((40, args.z_dim))
    fixed_y = torch.arange(10).repeat_interleave(4)
    args.callbacks = [ImageLoggingCallback(1000, fixed_noise, fixed_y)]

    kwargs = dict(img_size=32, img_depth=1)
    if args.conditional:
        condition_dim = 32
        condition_encoder = nn.Embedding(10, condition_dim)
        nn.init.normal_(condition_encoder.weight, 0, 0.02)
        kwargs.update(c_dim=condition_dim, c_encoder=condition_encoder)

    D = dcgan.Discriminator(depth_list=[64, 128, 256], **kwargs)
    G = dcgan.Generator(z_dim=args.z_dim, depth_list=[256, 128, 64], **kwargs)
    print(D)
    print(G)

    gan = GANSystem(D, G, **vars(args))
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(gan, dloader)


if __name__ == "__main__":
    main()
