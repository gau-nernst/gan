import argparse
from functools import partial

import numpy as np
import torch
import torchvision.datasets as TD
import torchvision.transforms as TT
from torch import Tensor, nn
from torch.utils.data import DataLoader

from modelling import get_discriminator_cls, get_generator_cls
from training import GANTrainer, GANTrainerConfig
from utils import add_args_from_cls, cls_from_args


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dcgan")
    parser.add_argument("--base_depth", type=int)
    parser.add_argument("--dataset", choices=["mnist", "celeba"])
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--n_steps", type=int, default=10_000)
    parser.add_argument("--n_log_imgs", type=int, default=40)
    parser.add_argument("--progressive_growing", action="store_true")
    parser.add_argument("--fade_duration", type=int)
    add_args_from_cls(parser, GANTrainerConfig)
    return parser


def build_model(args: argparse.Namespace):
    if args.dataset == "mnist":
        y_dim = 10
        y_encoder = nn.Embedding
        img_channels = 1

    elif args.dataset == "celeba":
        y_dim = 40
        y_encoder = partial(nn.Linear, bias=False)
        img_channels = 3

    kwargs = dict(img_size=args.img_size, img_channels=img_channels)

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
            kwargs.update(y_dim=y_dim, y_layer_factory=y_encoder)
        dis = d_cls(**kwargs)
        gen = g_cls(**kwargs, z_dim=args.z_dim)

    elif args.model in ("dcgan", "progressive_gan", "stylegan", "stylegan2"):
        if args.conditional:
            yemb_dim = 128
            dis = get_discriminator_cls("cgan")(
                d_cls(**kwargs, img_channels=kwargs["img_channels"] + yemb_dim),
                y_encoder(y_dim, yemb_dim),
            )
            gen = get_generator_cls("cgan")(
                g_cls(**kwargs, z_dim=args.z_dim + yemb_dim),
                y_encoder(y_dim, yemb_dim),
            )
        else:
            dis = d_cls(**kwargs)
            gen = g_cls(**kwargs)

    else:
        raise ValueError(f"Unsupported model {args.model}")

    return dis, gen


def build_dataset(name: str, img_size: int):
    if name == "mnist":
        transform = TT.Compose([TT.Pad(2), TT.ToTensor(), TT.Normalize(0.5, 0.5)])
        ds = TD.MNIST("data", transform=transform)

    elif name == "celeba":
        transform = TT.Compose([TT.Resize(img_size), TT.CenterCrop(img_size), TT.ToTensor(), TT.Normalize(0.5, 0.5)])
        ds = TD.CelebA("data", split="all", transform=transform, target_transform=lambda x: x / x.sum())

    else:
        raise ValueError(f"Dataset {name} is not supported")

    return ds


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.dataset == "celeba" and args.img_size > 256:
        print("CelebA image size is 178x218. Training large image size is not optimal")

    if args.dataset == "mnist":
        print("Only allow 32x32 image for MNIST")
        args.img_size = 32

    dis, gen = build_model(args)

    fixed_z = torch.randn((args.n_log_imgs, args.z_dim))

    ds = build_dataset(args.dataset, args.img_size)
    rand_indices = np.random.choice(len(ds), args.n_log_imgs, replace=False)
    fixed_y = [ds[idx][1] for idx in rand_indices]
    fixed_y = (torch.stack if isinstance(fixed_y[0], Tensor) else torch.tensor)(fixed_y)

    config = cls_from_args(args, GANTrainerConfig)
    trainer = GANTrainer(config, dis, gen, fixed_z, fixed_y)

    if args.progressive_growing:
        schedule = [(4, args.fade_duration)]
        while schedule[-1][0] != args.img_size:
            schedule.append((schedule[-1][0] * 2, args.fade_duration * 2))
    else:
        schedule = [(args.img_size, args.n_steps)]

    for i, (size, n_steps) in enumerate(schedule):
        if i > 0:
            dis.grow()
            gen.grow()

        dloader = DataLoader(
            dataset=build_dataset(args.dataset, size),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )
        trainer.train(dloader, n_steps)
    trainer.accelerator.end_training()


if __name__ == "__main__":
    main()
