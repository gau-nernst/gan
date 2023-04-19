import argparse
from functools import partial

import numpy as np
import torch
import torchvision.datasets as TD
import torchvision.transforms as TT
from torch import nn
from torch.utils.data import DataLoader

from modelling import get_discriminator_cls, get_generator_cls
from training import GANTrainer, GANTrainerConfig
from utils import cls_from_args, get_parser


def build_model(args: argparse.Namespace):
    y_dim = 40
    kwargs = dict(img_size=args.img_size, img_channels=3)

    if args.base_depth is not None:
        kwargs.update(base_depth=args.base_depth)

    if args.progressive_growing:
        assert args.model in ("progressive_gan",)
        kwargs.update(
            progressive_growing=args.progressive_growing,
            fade_duration=args.fade_duration,
        )

    d_cls = get_discriminator_cls(args.model)
    g_cls = get_generator_cls(args.model)

    if args.model == "sagan":
        if args.conditional:
            kwargs.update(y_dim=y_dim, y_layer_factory=partial(nn.Linear, bias=False))
        dis = d_cls(**kwargs)
        gen = g_cls(**kwargs, z_dim=args.z_dim)

    elif args.model in ("dcgan", "progressive_gan", "stylegan", "stylegan2"):
        if args.conditional:
            yemb_dim = 128
            dis = get_discriminator_cls("cgan")(
                d_cls(**kwargs, img_channels=kwargs["img_channels"] + yemb_dim),
                nn.Linear(y_dim, yemb_dim, bias=False),
            )
            gen = get_generator_cls("cgan")(
                g_cls(**kwargs, z_dim=args.z_dim + yemb_dim),
                nn.Linear(y_dim, yemb_dim, bias=False),
            )
        else:
            dis = d_cls(**kwargs)
            gen = g_cls(**kwargs)

    else:
        raise ValueError(f"Unsupported model {args.model}")

    return dis, gen


def build_dataloader(img_size: int, batch_size: int):
    transform = TT.Compose([TT.Resize(img_size), TT.CenterCrop(img_size), TT.ToTensor(), TT.Normalize(0.5, 0.5)])
    ds = TD.CelebA("data", split="all", transform=transform, target_transform=lambda x: x / x.sum())
    dloader = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    return dloader


def main():
    parser = get_parser()
    args = parser.parse_args()

    img_size = args.img_size
    assert img_size <= 256, "CelebA image size is 178x218. Cannot train GAN larger than 128x128"

    dis, gen = build_model(args)

    fixed_z = torch.randn((args.n_log_imgs, args.z_dim))

    fixed_y = TD.CelebA("data", split="all").attr
    fixed_y = fixed_y[torch.from_numpy(np.random.choice(fixed_y.shape[0], args.n_log_imgs, replace=False))]
    fixed_y = fixed_y / fixed_y.sum(1, keepdim=True)

    config = cls_from_args(args, GANTrainerConfig)
    trainer = GANTrainer(config, D, G, fixed_z, fixed_y)

    if args.progressive_growing:
        schedule = [(4, args.fade_duration)]
        while schedule[-1][0] != img_size:
            schedule.append((schedule[-1][0] * 2, args.fade_duration * 2))
    else:
        schedule = [(img_size, args.n_steps)]

    for i, (size, n_steps) in enumerate(schedule):
        if i > 0:
            dis.grow()
            gen.grow()
        dloader = build_dataloader(size, args.batch_size)
        trainer.train(dloader, n_steps)
    trainer.accelerator.end_training()


if __name__ == "__main__":
    main()
