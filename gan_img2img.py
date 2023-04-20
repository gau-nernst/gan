import argparse
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image

from data_utils import TorchStringArray
from modelling import PatchGAN, ResNetGenerator, UnetGenerator
from training import BaseTrainer, BaseTrainerConfig, compute_d_loss, compute_g_loss, log_images
from utils import add_args_from_cls, cls_from_args


# http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/
class Pix2PixDataset(Dataset):
    def __init__(self, data_dir: str):
        super().__init__()
        self.data_dir = data_dir
        files = os.listdir(self.data_dir)
        files.sort()
        self.files = TorchStringArray(files)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        img = read_image(os.path.join(self.data_dir, self.files[index]))
        img = img / 127.5 - 1
        imgA, imgB = torch.chunk(img, 2, 2)
        return imgA, imgB

    def __len__(self) -> int:
        return len(self.files)


@dataclass
class Img2ImgTrainerConfig(BaseTrainerConfig):
    swap_AB: bool = False
    l1_reg: float = 100.0


class Img2ImgTrainer(BaseTrainer):
    def __init__(self, config: Img2ImgTrainerConfig, dis: nn.Module, gen: nn.Module, fixed_imgs_A: Tensor):
        super().__init__(config, dis, gen)
        self.fixed_imgs_A = fixed_imgs_A.to(self.accelerator.device)

    def train_step(self, imgs_A: Tensor, imgs_B: Tensor):
        cfg = self.config
        if cfg.swap_AB:
            imgs_A, imgs_B = imgs_B, imgs_A

        if cfg.channels_last:
            imgs_A = imgs_A.to(memory_format=torch.channels_last)
            imgs_B = imgs_B.to(memory_format=torch.channels_last)

        if self.counter % cfg.log_img_interval == 0 and self.accelerator.is_main_process:
            with torch.inference_mode(), self.g_ema.swap_state_dict(self.gen):
                fake_imgs_B = self.gen(self.fixed_imgs_A)
            log_images(self.accelerator, fake_imgs_B, "generated", self.counter)

        with torch.no_grad():
            fake_imgs_B = self.gen(imgs_A)

        d_reals = self.dis(imgs_A, imgs_B)
        d_fakes = self.dis(imgs_A, fake_imgs_B)
        loss_d = compute_d_loss(d_reals, d_fakes, cfg.method)

        self.optim_d.zero_grad(set_to_none=True)
        self.accelerator.backward(loss_d)
        self.optim_d.step()

        self.dis.requires_grad_(False)
        fake_imgs_B = self.gen(imgs_A)
        d_fakes = self.dis(imgs_A, fake_imgs_B)
        loss_g = compute_g_loss(d_fakes, cfg.method) + F.l1_loss(fake_imgs_B, imgs_B) * cfg.l1_reg
        self.dis.requires_grad_(True)

        self.optim_g.zero_grad(set_to_none=True)
        self.accelerator.backward(loss_g)
        self.optim_g.step()

        return {"loss/d": loss_d.item(), "loss/g": loss_g.item()}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pix2pix", choices=["pix2pix", "cyclegan"])
    parser.add_argument("--dataset", choices=["edges2shoes", "facades", "maps", "night2day"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--n_steps", type=int, default=10_000)
    add_args_from_cls(parser, Img2ImgTrainerConfig)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    data_dir = f"../datasets/{args.dataset}/train"
    ds = Pix2PixDataset(data_dir)
    print(f"Image shape: {ds[0][0].shape}")

    dloader = DataLoader(
        ds,
        args.batch_size,
        True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    imgs_A, imgs_B = next(iter(dloader))

    dis = PatchGAN()
    gen = UnetGenerator() if args.model == "pix2pix" else ResNetGenerator()

    config = cls_from_args(args, Img2ImgTrainerConfig)
    trainer = Img2ImgTrainer(config, dis, gen, imgs_A)
    log_images(trainer.accelerator, imgs_A, "img/A", 0)
    log_images(trainer.accelerator, imgs_B, "img/B", 0)

    trainer.train(dloader, args.n_steps)


if __name__ == "__main__":
    main()
