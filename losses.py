import torch
import torch.nn.functional as F
from torch import Tensor, nn


def get_loss(name: str):
    return {
        "gan": GanLoss,
        "lsgan": LsGanLoss,
        "wgan": WganLoss,
        "wgan-gp": WganLoss,
        "hinge": HingeLoss,
        "relativistic-gan": RelativisticGanLoss,
    }[name]


def get_regularizer(name: str):
    return {
        "wgan-gp": wgan_gp_regularizer,
        "r1": r1_regularizer,
    }.get(name, None)


# https://arxiv.org/abs/1406.2661
class GanLoss:
    @staticmethod
    def d_loss(d_reals: Tensor, d_fakes: Tensor) -> Tensor:
        return -F.logsigmoid(d_reals).mean() - F.logsigmoid(-d_fakes).mean()

    @staticmethod
    def g_loss(d_fakes: Tensor, disc: nn.Module, reals: Tensor) -> Tensor:
        return -F.logsigmoid(d_fakes).mean()


# https://arxiv.org/abs/1611.04076
class LsGanLoss:
    @staticmethod
    def d_loss(d_reals: Tensor, d_fakes: Tensor) -> Tensor:
        return (d_reals - 1).square().mean() + d_fakes.square().mean()

    @staticmethod
    def g_loss(d_fakes: Tensor, disc: nn.Module, reals: Tensor) -> Tensor:
        return (d_fakes - 1).square().mean()


# https://arxiv.org/abs/1701.07875
# https://github.com/martinarjovsky/WassersteinGAN
class WganLoss:
    @staticmethod
    def d_loss(d_reals: Tensor, d_fakes: Tensor) -> Tensor:
        return -d_reals.mean() + d_fakes.mean()

    @staticmethod
    def g_loss(d_fakes: Tensor, disc: nn.Module, reals: Tensor) -> Tensor:
        return -d_fakes.mean()


# https://arxiv.org/abs/1802.05957
# https://github.com/pfnet-research/sngan_projection
# Although it is often cited as from https://arxiv.org/abs/1705.02894,
# the exact form used is from SNGAN
class HingeLoss:
    @staticmethod
    def d_loss(d_reals: Tensor, d_fakes: Tensor) -> Tensor:
        return F.relu(1 - d_reals).mean() + F.relu(1 + d_fakes).mean()

    g_loss = WganLoss.g_loss


# https://arxiv.org/abs/1807.00734
# https://github.com/AlexiaJM/RelativisticGAN
class RelativisticGanLoss:
    @staticmethod
    def d_loss(d_reals: Tensor, d_fakes: Tensor) -> Tensor:
        return -F.logsigmoid(d_reals - d_fakes).mean()

    @staticmethod
    def g_loss(d_fakes: Tensor, disc: nn.Module, reals: Tensor) -> Tensor:
        with torch.no_grad():
            d_reals = disc(reals)
        return -F.logsigmoid(d_fakes - d_reals).mean()


# https://arxiv.org/abs/1704.00028
# https://github.com/igul222/improved_wgan_training
# https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-penalty
def wgan_gp_regularizer(disc: nn.Module, reals: Tensor, fakes: Tensor, d_reals: Tensor) -> Tensor:
    alpha = torch.rand(reals.shape[0], 1, 1, 1, device=reals.device)
    inter = reals.lerp(fakes.detach().float(), alpha).requires_grad_()
    d_inter = disc(inter)

    with torch.autocast(reals.device.type, enabled=False):
        (d_grad,) = torch.autograd.grad(d_inter.sum(), inter, create_graph=True)

    d_grad_norm = torch.linalg.vector_norm(d_grad.flatten(1), dim=1)
    return (d_grad_norm - 1).square().mean() * 10


# https://arxiv.org/abs/1801.04406
def r1_regularizer(disc: nn.Module, reals: Tensor, fakes: Tensor, d_reals: Tensor) -> Tensor:
    with torch.autocast(reals.device.type, enabled=False):
        (d_grad,) = torch.autograd.grad(d_reals.sum(), reals, create_graph=True)

    d_grad_norm2 = d_grad.square().sum() / d_grad.shape[0]
    return d_grad_norm2 * 5
