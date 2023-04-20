import argparse
import os

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image

from data_utils import TorchStringArray
from modelling import PatchGAN, ResNetGenerator, UnetGenerator
from training import GANTrainer, GANTrainerConfig
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


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pix2pix", choice=["pix2pix", "cyclegan"])
    parser.add_argument("--base_depth", type=int)
    # parser.add_argument("--dataset", choices=["mnist", "celeba"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--n_steps", type=int, default=10_000)
    parser.add_argument("--n_log_imgs", type=int, default=40)
    add_args_from_cls(parser, GANTrainerConfig)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    data_dir = "../datasets/edges2shoes/train"
    ds = Pix2PixDataset(data_dir)

    dloader = DataLoader(
        ds,
        args.batch_size,
        True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    gen = UnetGenerator() if args.model == "pix2pix" else ResNetGenerator()
    dis = PatchGAN()

    config = cls_from_args(args, GANTrainerConfig)
    trainer = GANTrainer(config, dis, gen, None, None)


if __name__ == "__main__":
    main()
