import random
from copy import deepcopy
from functools import partial

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

    img_size = args.img_size
    transform = TT.Compose([TT.Resize(img_size), TT.CenterCrop(img_size), TT.ToTensor(), TT.Normalize(0.5, 0.5)])
    target_transform = lambda x: x / x.sum()
    ds = TD.CelebA("data", split="all", transform=transform, target_transform=target_transform)
    dloader = DataLoader(
        dataset=ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    y_dim = 40
    fixed_z = torch.randn((40, args.z_dim))
    fixed_y = torch.stack([ds[idx][1] for idx in random.sample(range(len(ds)), 40)], dim=0)

    d_kwargs = dict(img_size=img_size, img_depth=3)
    g_kwargs = dict(img_size=img_size, img_depth=3, z_dim=args.z_dim)
    if args.base_depth is not None:
        d_kwargs.update(base_depth=args.base_depth)
        g_kwargs.update(base_depth=args.base_depth)

    if args.model == "sagan":
        if args.conditional:
            d_kwargs.update(y_dim=y_dim, y_layer_factory=partial(nn.Linear, bias=False))
            g_kwargs.update(y_dim=y_dim, y_layer_factory=partial(nn.Linear, bias=False))
        D = modelling.sagan.Discriminator(**d_kwargs)
        G = modelling.sagan.Generator(**g_kwargs)

    elif args.model in ("dcgan", "progressive_gan", "stylegan", "stylegan2"):
        yemb_dim = 128
        if args.conditional:
            d_kwargs["img_depth"] += yemb_dim
            g_kwargs["z_dim"] += yemb_dim

        model = getattr(modelling, args.model)
        D = model.Discriminator(**d_kwargs)
        G = model.Generator(**g_kwargs)

        if args.conditional:
            y_encoder = nn.Linear(y_dim, yemb_dim, bias=False)
            nn.init.normal_(y_encoder.weight, 0, 0.02)

            D = modelling.cgan.Discriminator(D, y_encoder)
            G = modelling.cgan.Generator(G, deepcopy(y_encoder))

    else:
        raise ValueError(f"Unsupported model {args.model}")

    trainer = GANTrainer(config, D, G, fixed_z, fixed_y)
    trainer.train(dloader)


if __name__ == "__main__":
    main()
