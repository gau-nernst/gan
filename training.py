import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer


def _generate(G: nn.Module, z_dim: int, bsize: int, device):
    z_noise = torch.randn(bsize, z_dim, device=device)
    return G(z_noise)


def _optim_step(loss: torch.Tensor, optim: Optimizer):
    optim.zero_grad()
    loss.backward()
    optim.step()


def gan_train_step(
    D: nn.Module,
    G: nn.Module,
    x_reals: Tensor,
    z_dim: int,
    optim_d: Optimizer,
    optim_g: Optimizer,
    label_smoothing: float = 0.0,
):
    bsize = x_reals.size(0)
    device = x_reals.device

    # train D
    with torch.no_grad():
        x_fakes = _generate(G, z_dim, bsize, device)
    d_reals, d_fakes = map(D, (x_reals, x_fakes))
    loss_d_real = -F.logsigmoid(d_reals).mean() * (1.0 - label_smoothing)
    loss_d_fake = -F.logsigmoid(-d_fakes).mean()
    loss_d = loss_d_real + loss_d_fake
    _optim_step(loss_d, optim_d)

    # train G
    d_fakes = D(_generate(G, z_dim, bsize, device))
    loss_g = -F.logsigmoid(d_fakes).mean()
    _optim_step(loss_g, optim_g)

    return dict(loss_d=loss_d.item(), loss_g=loss_g.item())


def wgan_train_step(
    D: nn.Module,
    G: nn.Module,
    x_reals: Tensor,
    z_dim: int,
    optim_d: Optimizer,
    optim_g: Optimizer,
    weight_clipping: float = 0.01,
    train_g: bool = True,
):
    bsize = x_reals.size(0)
    device = x_reals.device

    # train D
    with torch.no_grad():
        x_fakes = _generate(G, z_dim, bsize, device)
    d_reals, d_fakes = map(D, (x_reals, x_fakes))
    loss_d = d_fakes.mean() - d_reals.mean()
    _optim_step(loss_d, optim_d)
    loss_dict = dict(loss_d=loss_d.item())

    with torch.no_grad():
        for param in D.parameters():
            param.clip_(-weight_clipping, weight_clipping)

    # train G
    if train_g:
        optim_g.zero_grad()
        loss_g = -D(_generate(G, z_dim, bsize, device)).mean()
        _optim_step(loss_g, optim_g)
        loss_dict.update(loss_g=loss_g.item())

    return loss_dict


def wgangp_train_step(
    D: nn.Module,
    G: nn.Module,
    x_reals: Tensor,
    z_dim: int,
    optim_d: Optimizer,
    optim_g: Optimizer,
    lamb: float = 10.0,
    train_g: bool = True,
):
    bsize = x_reals.size(0)
    device = x_reals.device

    # train D
    with torch.no_grad():
        x_fakes = _generate(G, z_dim, bsize, device)
    d_reals, d_fakes = map(D, (x_reals, x_fakes))
    loss_d = d_fakes.mean() - d_reals.mean()

    alpha = torch.rand(bsize, 1, 1, 1, device=device)
    x_inters = (x_reals * alpha + x_fakes * (1 - alpha)).requires_grad_()
    d_inters = D(x_inters)

    d_grad = torch.autograd.grad(d_inters, x_inters, torch.ones_like(d_inters), create_graph=True)[0]
    d_grad_norm = torch.linalg.vector_norm(d_grad, dim=(1, 2, 3))
    loss_d = loss_d + lamb * ((d_grad_norm - 1) ** 2).mean()
    
    _optim_step(loss_d, optim_d)
    loss_dict = dict(loss_d=loss_d.item())

    # train G
    if train_g:
        optim_g.zero_grad()
        loss_g = -D(_generate(G, z_dim, bsize, device)).mean()
        _optim_step(loss_g, optim_g)
        loss_dict.update(loss_g=loss_g.item())

    return loss_dict
