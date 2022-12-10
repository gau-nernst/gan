import datetime

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

    fixed_noise = torch.randn((30, args.z_dim))
    fixed_y = torch.arange(10).repeat_interleave(3)
    args.callbacks = [ImageLoggingCallback(1000, fixed_noise, fixed_y)]

    D = dcgan.Discriminator(32, 1, [64, 128, 256], c_dim=32)
    G = dcgan.Generator(32, 1, args.z_dim, [256, 128, 64], c_dim=32)
    condition_encoder = nn.Embedding(10, 32)
    nn.init.normal_(condition_encoder.weight, 0, 0.02)
    print(D)
    print(G)

    gan = GANSystem(D, G, condition_encoder=condition_encoder, **vars(args))
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(gan, dloader)


if __name__ == "__main__":
    main()
