import torch
import torch.nn.functional as F
from torch import Tensor, nn


def get_gan_loss(name: str):
    return {
        "gan": GAN,
        "wgan": WGAN,
        "rgan": RGAN,
    }[name]


class GAN:
    @staticmethod
    def d_loss(disc: nn.Module, reals: Tensor, fakes: Tensor) -> Tensor:
        return -F.logsigmoid(disc(reals)).mean() - F.logsigmoid(-disc(fakes)).mean()

    @staticmethod
    def g_loss(d_fakes: Tensor, disc: nn.Module, reals: Tensor) -> Tensor:
        return -F.logsigmoid(d_fakes).mean()


class WGAN:
    @staticmethod
    def d_loss(disc: nn.Module, reals: Tensor, fakes: Tensor) -> Tensor:
        with torch.no_grad():
            for p in disc.parameters():
                p.clip_(-0.01, 0.01)
        return -disc(reals).mean() + disc(fakes).mean()

    @staticmethod
    def g_loss(d_fakes: Tensor, disc: nn.Module, reals: Tensor) -> Tensor:
        return -d_fakes.mean()


class RGAN:
    @staticmethod
    def d_loss(disc: nn.Module, reals: Tensor, fakes: Tensor) -> Tensor:
        return -F.logsigmoid(disc(reals) - disc(fakes)).mean()

    @staticmethod
    def g_loss(d_fakes: Tensor, disc: nn.Module, reals: Tensor) -> Tensor:
        with torch.no_grad():
            d_reals = disc(reals)
        return -F.logsigmoid(d_fakes - d_reals).mean()
