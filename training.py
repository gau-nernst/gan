import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer


def train_step(
    D: nn.Module,
    G: nn.Module,
    x_reals: Tensor,
    z_dim: int,
    optim_d: Optimizer,
    optim_g: Optimizer
):
    bsize = x_reals.size(0)
    device = x_reals.device

    # train D
    z_noise = torch.randn(bsize, z_dim, device=device)
    with torch.no_grad():
        x_fakes = G(z_noise)
    d_reals, d_fakes = map(D, (x_reals, x_fakes))
    loss_d = -(F.logsigmoid(d_reals).mean() + F.logsigmoid(-d_fakes).mean())
    
    optim_d.zero_grad()
    loss_d.backward()
    optim_d.step()

    # train G
    z_noise = torch.randn(bsize, z_dim, device=device)
    x_fakes = G(z_noise)
    d_fakes = D(x_fakes)
    loss_g = -F.logsigmoid(d_fakes).mean()
    
    optim_g.zero_grad()
    loss_g.backward()
    optim_g.step()

    return dict(loss_d=loss_d.item(), loss_g=loss_g.item())
