import torch
import torch.nn.functional as F
from torch import Tensor, nn


def get_gan_loss(name: str):
    return {
        "gan": GAN,
        "wgan": WGAN,
        "wgan-gp": WGAN_GP,
        "rgan": RGAN,
    }[name]


# https://arxiv.org/abs/1406.2661
class GAN:
    @staticmethod
    def d_loss(disc: nn.Module, reals: Tensor, fakes: Tensor) -> Tensor:
        return -F.logsigmoid(disc(reals)).mean() - F.logsigmoid(-disc(fakes)).mean()

    @staticmethod
    def g_loss(d_fakes: Tensor, disc: nn.Module, reals: Tensor) -> Tensor:
        return -F.logsigmoid(d_fakes).mean()


# https://arxiv.org/abs/1701.07875
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


# https://arxiv.org/abs/1704.00028
# https://github.com/igul222/improved_wgan_training
class WGAN_GP:
    @staticmethod
    def d_loss(disc: nn.Module, reals: Tensor, fakes: Tensor) -> Tensor:
        loss_d = -disc(reals).mean() + disc(fakes).mean()

        # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-penalty
        alpha = torch.rand(reals.shape[0], 1, 1, 1, device=reals.device)
        interpolates = reals.lerp(fakes.detach(), alpha).requires_grad_()
        d_interpolates = disc(interpolates)

        with torch.autocast(enabled=False):
            (d_grad,) = torch.autograd.grad(d_interpolates.sum(), interpolates, create_graph=True)

        d_grad_norm = torch.linalg.vector_norm(d_grad.flatten(1), dim=1)
        return loss_d + (d_grad_norm - 1).square().mean() * 10

    @staticmethod
    def g_loss(d_fakes: Tensor, disc: nn.Module, reals: Tensor) -> Tensor:
        return -d_fakes.mean()


# https://arxiv.org/abs/1807.00734
class RGAN:
    @staticmethod
    def d_loss(disc: nn.Module, reals: Tensor, fakes: Tensor) -> Tensor:
        return -F.logsigmoid(disc(reals) - disc(fakes)).mean()

    @staticmethod
    def g_loss(d_fakes: Tensor, disc: nn.Module, reals: Tensor) -> Tensor:
        with torch.no_grad():
            d_reals = disc(reals)
        return -F.logsigmoid(d_fakes - d_reals).mean()
