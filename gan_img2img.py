import argparse
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from datasets import ImageFolderDataset
from modelling import PatchGAN, ResNetGenerator, UnetGenerator
from training import BaseTrainer, BaseTrainerConfig, compute_d_loss, compute_g_loss, log_images
from utils import add_args_from_cls, cls_from_args


# http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/data/aligned_dataset.py
class AlignedDataset(ImageFolderDataset):
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return torch.chunk(super().__getitem__(idx), 2, 2)


# http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/data/aligned_dataset.py
class UnalignedDataset(Dataset):
    def __init__(self, data_dir_A: str, data_dir_B: str):
        super().__init__()
        self.ds_A = ImageFolderDataset(data_dir_A)
        self.ds_B = ImageFolderDataset(data_dir_B)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.ds_A[idx], self.ds_B[torch.randint(len(self.ds_B), ()).item()]

    def __len__(self):
        return len(self.ds_A)


@dataclass
class Img2ImgTrainerConfig(BaseTrainerConfig):
    model: Literal["pix2pix", "cyclegan"] = "pix2pix"
    swap_AB: bool = False
    l1_reg: float = 0.0


class Img2ImgTrainer(BaseTrainer):
    def __init__(
        self, config: Img2ImgTrainerConfig, dis: nn.Module, gen: nn.Module, fixed_imgs_A: Tensor, fixed_imgs_B: Tensor
    ):
        super().__init__(config, dis, gen)
        self.fixed_imgs_A = fixed_imgs_A.to(self.accelerator.device)
        self.fixed_imgs_B = fixed_imgs_B.to(self.accelerator.device)

    @staticmethod
    def d_loss(gen_AB, dis_B, imgs_A, imgs_B, method):
        with torch.no_grad():
            fake_imgs_B = gen_AB(imgs_A)

        d_reals = dis_B(imgs_A, imgs_B)
        d_fakes = dis_B(imgs_A, fake_imgs_B)
        return compute_d_loss(d_reals, d_fakes, method)

    @staticmethod
    def g_loss(gen_AB, dis_B, imgs_A, imgs_B, method, l1_reg, gen_BA=None):
        dis_B.requires_grad_(False)
        fake_imgs_B = gen_AB(imgs_A)
        d_fakes = dis_B(imgs_A, fake_imgs_B)
        loss_g = compute_g_loss(d_fakes, method)

        if l1_reg > 0:
            loss_g = loss_g + F.l1_loss(fake_imgs_B, imgs_B) * l1_reg

        if gen_BA is not None:
            fake_fake_imgs_A = gen_BA(fake_imgs_B)
            loss_g = loss_g + F.l1_loss(fake_fake_imgs_A, imgs_A) * 10

        dis_B.requires_grad_(True)
        return loss_g

    def train_step(self, imgs_A: Tensor, imgs_B: Tensor):
        cfg = self.config
        if cfg.swap_AB:
            imgs_A, imgs_B = imgs_B, imgs_A

        if cfg.channels_last:
            imgs_A = imgs_A.to(memory_format=torch.channels_last)
            imgs_B = imgs_B.to(memory_format=torch.channels_last)

        gen_AB, gen_BA = self.gen[0], None
        dis_B, dis_A = self.dis[0], None
        if cfg.model == "cyclegan":
            gen_BA = self.gen[1]
            dis_A = self.dis[1]

        if self.counter % cfg.log_img_interval == 0 and self.accelerator.is_main_process:
            with torch.inference_mode(), self.g_ema.swap_state_dict(self.gen):
                fake_imgs_B = gen_AB(self.fixed_imgs_A)
                log_images(self.accelerator, fake_imgs_B, "generated/B", self.counter)

                if gen_BA is not None:
                    fake_imgs_A = gen_BA(self.fixed_imgs_B)
                    log_images(self.accelerator, fake_imgs_A, "generated/A", self.counter)

        loss_d = self.d_loss(gen_AB, dis_B, imgs_A, imgs_B, cfg.method)

        if gen_BA is not None:
            loss_d = loss_d + self.d_loss(gen_BA, dis_A, imgs_B, imgs_A, cfg.method)

        self.optim_d.zero_grad(set_to_none=True)
        self.accelerator.backward(loss_d)
        self.optim_d.step()

        loss_g = self.g_loss(gen_AB, dis_B, imgs_A, imgs_B, cfg.method, cfg.l1_reg, gen_BA)

        if cfg.model == "cyclegan":
            loss_g = loss_g + self.g_loss(gen_BA, dis_A, imgs_B, imgs_A, cfg.method, cfg.l1_reg, gen_AB)

        self.optim_g.zero_grad(set_to_none=True)
        self.accelerator.backward(loss_g)
        self.optim_g.step()

        return {"loss/d": loss_d.item(), "loss/g": loss_g.item()}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--n_steps", type=int, default=10_000)
    parser.add_argument("--n_log_imgs", type=int, default=16)
    parser.add_argument("--compile", action="store_true")
    add_args_from_cls(parser, Img2ImgTrainerConfig)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.model == "pix2pix":
        dis = [PatchGAN()]
        gen = [UnetGenerator()]
        train_ds = AlignedDataset(f"../datasets/{args.dataset}/train")
        val_ds = AlignedDataset(f"../datasets/{args.dataset}/val")

    elif args.model == "cyclegan":
        # this trick does not work with torch.compile()
        dis = [PatchGAN(), PatchGAN()]
        gen = [ResNetGenerator(), ResNetGenerator()]
        train_ds = UnalignedDataset(f"../datasets/{args.dataset}/trainA", f"../datasets/{args.dataset}/trainB")
        val_ds = UnalignedDataset(f"../datasets/{args.dataset}/testA", f"../datasets/{args.dataset}/testB")

    else:
        raise ValueError(f"Model {args.model} is not supported")

    fn = torch.compile if args.compile else lambda x: x
    dis = nn.ModuleList(fn(x) for x in dis)
    gen = nn.ModuleList(fn(x) for x in gen)

    print(f"Image shape: {train_ds[0][0].shape}")

    rand_indices = np.random.choice(len(val_ds), args.n_log_imgs, replace=False)
    imgs_A, imgs_B = zip(*[val_ds[idx] for idx in rand_indices])
    imgs_A = torch.stack(imgs_A)
    imgs_B = torch.stack(imgs_B)

    config = cls_from_args(args, Img2ImgTrainerConfig)
    trainer = Img2ImgTrainer(config, dis, gen, imgs_A, imgs_B)
    log_images(trainer.accelerator, imgs_A, "real/A", 0)
    log_images(trainer.accelerator, imgs_B, "real/B", 0)

    dloader = DataLoader(
        train_ds,
        args.batch_size,
        True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    trainer.train(dloader, args.n_steps)


if __name__ == "__main__":
    main()
