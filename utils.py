import argparse
from typing import Callable

from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.utils import make_grid

_Norm = Callable[[int], nn.Module]
_Act = Callable[[], nn.Module]


def repeat_dataloader(dataloader):
    while True:
        for data in dataloader:
            yield data


class TensorboardWriter(SummaryWriter):
    def add_images(self, tag, images, *args, nrow=8, **kwargs):
        grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
        self.add_image(tag, grid, *args, **kwargs)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["gan", "wgan", "wgan-gp"], default="gan")
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--log_interval", type=int, default=2_000)
    parser.add_argument("--log_comment", default="")
    return parser
