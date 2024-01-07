# https://arxiv.org/abs/2006.10738
# https://github.com/mit-han-lab/data-efficient-gans

import torch
from torch import Tensor, nn
from torchvision.transforms import v2


def rand_brightness(x: Tensor) -> Tensor:
    return x + torch.rand(*x.shape[:-3], 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5


def rand_saturation(x: Tensor) -> Tensor:
    mean = x.mean(dim=-3, keepdim=True)
    return mean + (x - mean) * torch.rand(*x.shape[:-3], 1, 1, 1, dtype=x.dtype, device=x.device) * 2


def rand_contrast(x: Tensor) -> Tensor:
    mean = x.mean(dim=[-1, -2, -3], keepdim=True)
    return mean + (x - mean) * (torch.rand(*x.shape[:-3], 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5)


# v2.ColorJitter will clamp the output to [0,1], so we cannot use it
class ColorJitter(nn.Module):
    def forward(self, x: Tensor):
        x = rand_brightness(x)
        x = rand_saturation(x)
        x = rand_contrast(x)
        return x


class DiffAugment(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            ColorJitter(),
            v2.RandomAffine(0, (0.125, 0.125)),
            v2.RandomErasing(p=1.0),  # not exactly the same
        )
