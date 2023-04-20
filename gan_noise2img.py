import argparse
from functools import partial
from typing import Optional

import numpy as np
import torch
import torchvision.datasets as TD
import torchvision.transforms as TT
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from modelling import get_discriminator_cls, get_generator_cls
from training import BaseTrainer, BaseTrainerConfig, compute_d_loss, compute_g_loss, cycle, log_images
from utils import add_args_from_cls, cls_from_args


def build_model(args: argparse.Namespace):
    if args.dataset == "mnist":
        y_dim = 10
        y_encoder = nn.Embedding
        img_channels = 1

    elif args.dataset == "celeba":
        y_dim = 40
        y_encoder = partial(nn.Linear, bias=False)
        img_channels = 3

    else:
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
        transform = TT.Compose([TT.Resize(img_size), TT.CenterCrop(img_size), TT.ToTensor(), TT.Normalize(0.5, 0.5)])
        ds = TD.ImageFolder(name, transform=transform)

    return ds


class Noise2ImgTrainer(BaseTrainer):
    def __init__(self, config, dis, gen, fixed_z, fixed_y):
        super().__init__(config, dis, gen)
        self.fixed_z = fixed_z.to(self.accelerator.device)
        self.fixed_y = fixed_y.to(self.accelerator.device)

    def _forward(self, m: nn.Module, xs: Tensor, ys: Optional[Tensor]) -> Tensor:
        return m(xs, ys) if self.config.conditional else m(xs)

    def train_step(self, x_reals, ys):
        cfg = self.config
        if cfg.channels_last:
            x_reals = x_reals.to(memory_format=torch.channels_last)
            if ys.dim() == 4:
                ys = ys.to(torch.channels_last)

        if self.counter % cfg.log_img_interval == 0 and self.accelerator.is_main_process:
            with torch.inference_mode(), self.g_ema.swap_state_dict(self.gen):
                x_fakes = self._forward(self.gen, self.fixed_z, self.fixed_y)
            log_images(self.accelerator, x_fakes, "generated", self.counter)

        log_dict = dict()
        log_dict["loss/d"] = self.train_d_step(x_reals, ys).item()

        if self.counter % cfg.train_g_interval == 0:
            log_dict["loss/g"] = self.train_g_step(x_reals, ys).item()
            self.g_ema.update(self.gen)

        for m in (self.dis, self.gen):
            m = m.module if self.accelerator.state.distributed_type == "MULTI_GPU" else m
            if hasattr(m, "step"):
                m.step()

        return log_dict

    def train_d_step(self, x_reals: Tensor, ys: Optional[Tensor]):
        bsize = x_reals.shape[0]
        cfg = self.config

        # Algorithm 1 in paper clip weights after optimizer step, but GitHub code clip before optimizer step
        # it shouldn't matter much in practice
        if cfg.method == "wgan":
            with torch.no_grad():
                for param in self.dis.parameters():
                    param.clip_(-cfg.wgan_clip, cfg.wgan_clip)

        with torch.no_grad():
            z_noise = torch.randn(bsize, cfg.z_dim, device=self.accelerator.device)
            x_fakes = self._forward(self.gen, z_noise, ys)

        if cfg.r1_penalty > 0 and self.counter % cfg.r1_penalty_interval == 0:
            x_reals.requires_grad_()

        d_reals = self._forward(self.dis, x_reals, ys)
        d_fakes = self._forward(self.dis, x_fakes, ys)

        loss_d = compute_d_loss(d_reals, d_fakes, cfg.method)

        if cfg.method == "wgan-gp":
            alpha = torch.rand(bsize, 1, 1, 1, device=x_reals.device)
            x_inters = x_reals.detach().lerp(x_fakes.detach(), alpha).requires_grad_()
            d_inters = self._forward(self.dis, x_inters, ys)

            (d_grad,) = torch.autograd.grad(d_inters.sum(), x_inters, create_graph=True)
            d_grad_norm = torch.linalg.vector_norm(d_grad, dim=(1, 2, 3))
            loss_d = loss_d + cfg.wgan_gp_lamb * (d_grad_norm - 1).square().mean()

        # for Progressive GAN only
        if cfg.drift_penalty > 0:
            loss_d = loss_d + d_reals.square().mean() * cfg.drift_penalty

        # https://arxiv.org/abs/1801.04406, for StyleGAN and StyleGAN2
        if cfg.r1_penalty > 0 and self.counter % cfg.r1_penalty_interval == 0:
            (d_grad,) = torch.autograd.grad(d_reals.sum(), x_reals, create_graph=True)
            d_grad_norm2 = d_grad.square().sum() / bsize
            loss_d = loss_d + d_grad_norm2 * cfg.r1_penalty / 2

        self.optim_d.zero_grad(set_to_none=True)
        self.accelerator.backward(loss_d)
        self.optim_d.step()

        return loss_d

    def train_g_step(self, x_reals: Tensor, ys: Optional[Tensor]):
        bsize = x_reals.shape[0]
        cfg = self.config
        self.dis.requires_grad_(False)

        z_noise = torch.randn(bsize, cfg.z_dim, device=self.accelerator.device)
        x_fakes = self._forward(self.gen, z_noise, ys)
        d_fakes = self._forward(self.dis, x_fakes, ys)

        loss_g = compute_g_loss(d_fakes, cfg.method)
        self.optim_g.zero_grad(set_to_none=True)
        self.accelerator.backward(loss_g)
        self.optim_g.step()

        self.dis.requires_grad_(True)
        return loss_g


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
    add_args_from_cls(parser, BaseTrainerConfig)
    return parser


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

    config = cls_from_args(args, BaseTrainerConfig)
    trainer = Noise2ImgTrainer(config, dis, gen, fixed_z, fixed_y)

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
