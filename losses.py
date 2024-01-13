import torch
import torch.nn.functional as F
from torch import Tensor, nn


def get_gan_loss(name: str):
    return {
        "gan": GAN,
        "lsgan": LSGAN,
        "wgan": WGAN,
        "wgan-gp": WGAN_GP,
        "hinge": HingeLoss,
        "rgan": RGAN,
    }[name]


# https://arxiv.org/abs/1406.2661
class GAN:
    @staticmethod
    def d_loss(disc: nn.Module, reals: Tensor, fakes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        d_reals, d_fakes = disc(reals), disc(fakes)
        return -F.logsigmoid(d_reals).mean() - F.logsigmoid(-d_fakes).mean(), d_reals, d_fakes

    @staticmethod
    def g_loss(d_fakes: Tensor, disc: nn.Module, reals: Tensor) -> Tensor:
        return -F.logsigmoid(d_fakes).mean()


# https://arxiv.org/abs/1611.04076
class LSGAN:
    @staticmethod
    def d_loss(disc: nn.Module, reals: Tensor, fakes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        d_reals, d_fakes = disc(reals), disc(fakes)
        return (d_reals - 1).square().mean() + d_fakes.square().mean(), d_reals, d_fakes

    @staticmethod
    def g_loss(d_fakes: Tensor, disc: nn.Module, reals: Tensor) -> Tensor:
        return (d_fakes - 1).square().mean()


# https://arxiv.org/abs/1701.07875
# https://github.com/martinarjovsky/WassersteinGAN
class WGAN:
    @staticmethod
    def d_loss(disc: nn.Module, reals: Tensor, fakes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        with torch.no_grad():
            for p in disc.parameters():
                p.clip_(-0.01, 0.01)
        d_reals, d_fakes = disc(reals), disc(fakes)
        return -d_reals.mean() + d_fakes.mean(), d_reals, d_fakes

    @staticmethod
    def g_loss(d_fakes: Tensor, disc: nn.Module, reals: Tensor) -> Tensor:
        return -d_fakes.mean()


# https://arxiv.org/abs/1704.00028
# https://github.com/igul222/improved_wgan_training
class WGAN_GP:
    @staticmethod
    def d_loss(disc: nn.Module, reals: Tensor, fakes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        d_reals, d_fakes = disc(reals), disc(fakes)
        loss_d = -d_reals.mean() + d_fakes.mean()

        # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-penalty
        alpha = torch.rand(reals.shape[0], 1, 1, 1, device=reals.device)
        interpolates = reals.lerp(fakes.detach().float(), alpha).requires_grad_()
        d_interpolates = disc(interpolates)

        with torch.autocast(reals.device.type, enabled=False):
            (d_grad,) = torch.autograd.grad(d_interpolates.sum(), interpolates, create_graph=True)

        d_grad_norm = torch.linalg.vector_norm(d_grad.flatten(1), dim=1)
        return loss_d + (d_grad_norm - 1).square().mean() * 10, d_reals, d_fakes

    g_loss = WGAN.g_loss


# https://arxiv.org/abs/1802.05957
# https://github.com/pfnet-research/sngan_projection
# Although it is often cited as from https://arxiv.org/abs/1705.02894,
# the exact form used is from SNGAN
class HingeLoss:
    @staticmethod
    def d_loss(disc: nn.Module, reals: Tensor, fakes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        d_reals, d_fakes = disc(reals), disc(fakes)
        return F.relu(1 - d_reals).mean() + F.relu(1 + d_fakes).mean(), d_reals, d_fakes

    g_loss = WGAN.g_loss


# https://arxiv.org/abs/1807.00734
# https://github.com/AlexiaJM/RelativisticGAN
class RGAN:
    @staticmethod
    def d_loss(disc: nn.Module, reals: Tensor, fakes: Tensor) -> Tensor:
        d_reals, d_fakes = disc(reals), disc(fakes)
        return -F.logsigmoid(d_reals - d_fakes).mean(), d_reals, d_fakes

    @staticmethod
    def g_loss(d_fakes: Tensor, disc: nn.Module, reals: Tensor) -> Tensor:
        with torch.no_grad():
            d_reals = disc(reals)
        return -F.logsigmoid(d_fakes - d_reals).mean()
