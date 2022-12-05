import pytorch_lightning as pl
import torchvision.datasets as TD
import torchvision.transforms as TT
from torch import nn
from torch.utils.data import DataLoader

import modelling
from training import GANSystem
from utils import get_parser


def main():
    parser = get_parser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    transform = TT.Compose([TT.Resize(64), TT.CenterCrop(64), TT.ToTensor(), TT.Normalize(0.5, 0.5)])
    ds = TD.CelebA("data", split="all", transform=transform)
    dloader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # DCGAN
    D = modelling.dcgan.Discriminator()
    G = modelling.dcgan.Generator(args.z_dim)

    gan = GANSystem(D, G, **vars(args))
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(gan, dloader)


if __name__ == "__main__":
    main()
