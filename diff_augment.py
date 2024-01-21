# https://arxiv.org/abs/2006.10738
# https://github.com/mit-han-lab/data-efficient-gans

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def rand_int(lo: int, hi: int) -> int:
    return torch.randint(lo, hi, ()).item()


def rand_brightness(x: Tensor) -> Tensor:
    return x + torch.rand(*x.shape[:-3], 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5


def rand_saturation(x: Tensor) -> Tensor:
    mean = x.mean(dim=-3, keepdim=True)
    return mean + (x - mean) * torch.rand(*x.shape[:-3], 1, 1, 1, dtype=x.dtype, device=x.device) * 2


def rand_contrast(x: Tensor) -> Tensor:
    mean = x.mean(dim=[-1, -2, -3], keepdim=True)
    return mean + (x - mean) * (torch.rand(*x.shape[:-3], 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5)


class ColorJitter(nn.Module):
    def forward(self, x: Tensor):
        x = rand_brightness(x)
        x = rand_saturation(x)
        x = rand_contrast(x)
        return x


def translate(x: Tensor, offset_x: int, offset_y: int) -> Tensor:
    return F.pad(x, (offset_x, -offset_x, offset_y, -offset_y))


class RandomTranslate(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        H, W = x.shape[-2:]
        pad_h = H // 8
        pad_w = W // 8
        offset_y = rand_int(-pad_h, pad_h + 1)
        offset_x = rand_int(-pad_w, pad_w + 1)
        return translate(x, offset_x, offset_y)


class RandomCutout(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        H, W = x.shape[-2:]
        offset_y = rand_int(0, H // 2)
        offset_x = rand_int(0, W // 2)
        x[..., offset_y : offset_y + H // 2, offset_x : offset_x + H // 2] = 0
        return x


class DiffAugment(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            ColorJitter(),
            RandomTranslate(),
            RandomCutout(),
        )
