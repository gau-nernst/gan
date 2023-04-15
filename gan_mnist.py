from copy import deepcopy

import torch
import torchvision.datasets as TD
import torchvision.transforms as TT
from torch import nn
from torch.utils.data import DataLoader

import modelling
from training import GANTrainer, GANTrainerConfig
from utils import cls_from_args, get_parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    config = cls_from_args(args, GANTrainerConfig)

    # pad 28x28 to 32x32
    transform = TT.Compose([TT.Pad(2), TT.ToTensor(), TT.Normalize(0.5, 0.5)])
    ds = TD.MNIST("data", transform=transform)
    dloader = DataLoader(
        dataset=ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    y_dim = 10
    n_log_imgs = args.n_log_imgs // 10 * 10
    fixed_z = torch.randn((n_log_imgs, args.z_dim))
    fixed_y = torch.arange(y_dim).repeat_interleave(n_log_imgs // y_dim)

    d_kwargs = dict(img_size=32, img_channels=1)
    g_kwargs = dict(img_size=32, img_channels=1, z_dim=args.z_dim)

    if args.model == "sagan":
        if args.conditional:
            d_kwargs.update(y_dim=y_dim)
            g_kwargs.update(y_dim=y_dim)
        D = modelling.sagan.Discriminator(**d_kwargs)
        G = modelling.sagan.Generator(**g_kwargs)

    elif args.model in ("dcgan", "progressive_gan", "stylegan", "stylegan2"):
        yemb_dim = 64
        if args.conditional:
            d_kwargs["img_channels"] += yemb_dim
            g_kwargs["z_dim"] += yemb_dim

        model = getattr(modelling, args.model)
        D = model.Discriminator(**d_kwargs)
        G = model.Generator(**g_kwargs)

        if args.conditional:
            y_encoder = nn.Embedding(y_dim, yemb_dim)
            nn.init.normal_(y_encoder.weight, 0, 0.02)

            D = modelling.cgan.Discriminator(D, y_encoder)
            G = modelling.cgan.Generator(G, deepcopy(y_encoder))

    else:
        raise ValueError(f"Unsupported model {args.model}")

    trainer = GANTrainer(config, D, G, fixed_z, fixed_y)
    trainer.train(dloader)


if __name__ == "__main__":
    main()
