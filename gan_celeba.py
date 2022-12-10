import datetime

import pytorch_lightning as pl
import torchvision.datasets as TD
import torchvision.transforms as TT
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.argparse import add_argparse_args
from torch.utils.data import DataLoader

from modelling import dcgan
from training import GANSystem
from utils import get_parser


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
    dloader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.logger = TensorBoardLogger("logs", name=args.log_name, version=timestamp)

    # DCGAN
    D = dcgan.Discriminator(img_size=img_size)
    G = dcgan.Generator(img_size=img_size, z_dim=args.z_dim)
    print(D)
    print(G)

    gan = GANSystem(D, G, **vars(args))
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(gan, dloader)


if __name__ == "__main__":
    main()
