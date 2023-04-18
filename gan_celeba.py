from functools import partial

import numpy as np
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
    assert img_size <= 256, "CelebA image size is 178x218. Cannot train GAN larger than 128x128"

    y_dim = 40
    fixed_z = torch.randn((args.n_log_imgs, args.z_dim))

    fixed_y = TD.CelebA("data", split="all").attr
    fixed_y = fixed_y[torch.from_numpy(np.random.choice(fixed_y.shape[0], args.n_log_imgs, replace=False))]
    fixed_y = fixed_y / fixed_y.sum(1, keepdim=True)

    kwargs = dict(img_size=img_size, img_channels=3)

    if args.base_depth is not None:
        kwargs.update(base_depth=args.base_depth)

    if args.progressive_growing:
        assert args.model in ("progressive_gan",)
        kwargs.update(
            progressive_growing=args.progressive_growing,
            fade_duration=args.fade_duration,
        )

    if args.model == "sagan":
        if args.conditional:
            kwargs.update(y_dim=y_dim, y_layer_factory=partial(nn.Linear, bias=False))
        D = modelling.sagan.Discriminator(**kwargs)
        G = modelling.sagan.Generator(**kwargs, z_dim=args.z_dim)

    elif args.model in ("dcgan", "progressive_gan", "stylegan", "stylegan2"):
        model = getattr(modelling, args.model)

        if args.conditional:
            yemb_dim = 128
            D = modelling.cgan.Discriminator(
                model.Discriminator(**kwargs, img_channels=kwargs["img_channels"] + yemb_dim),
                nn.Linear(y_dim, yemb_dim, bias=False),
            )
            G = modelling.cgan.Generator(
                model.Generator(**kwargs, z_dim=args.z_dim + yemb_dim),
                nn.Linear(y_dim, yemb_dim, bias=False),
            )
        else:
            D = model.Discriminator(**kwargs)
            G = model.Generator(**kwargs)

    else:
        raise ValueError(f"Unsupported model {args.model}")

    trainer = GANTrainer(config, D, G, fixed_z, fixed_y)

    if args.progressive_growing:
        schedule = [(4, args.fade_duration)]
        while schedule[-1][0] != img_size:
            schedule.append((schedule[-1][0] * 2, args.fade_duration * 2))
    else:
        schedule = [(img_size, args.n_steps)]

    for i, (size, n_steps) in enumerate(schedule):
        if i > 0:
            D.grow()
            G.grow()
        transform = TT.Compose([TT.Resize(size), TT.CenterCrop(size), TT.ToTensor(), TT.Normalize(0.5, 0.5)])
        ds = TD.CelebA("data", split="all", transform=transform, target_transform=lambda x: x / x.sum())
        dloader = DataLoader(
            dataset=ds,
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
