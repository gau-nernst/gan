import pytest
import torch
import torch.nn.functional as F

from modelling import dcgan, progressive_gan, sagan, stylegan, stylegan2
from modelling.nvidia_ops import Blur, UpFIRDn2d, upfirdn2d


IMG_SIZE = 64
IMG_DEPTH = 3
Z_DIM = 128
BATCH_SIZE = 4
IMG_SHAPE = (BATCH_SIZE, IMG_DEPTH, IMG_SIZE, IMG_SIZE)
NOISE_SHAPE = (BATCH_SIZE, Z_DIM)


@pytest.mark.parametrize("module", (dcgan, progressive_gan, stylegan, stylegan2, sagan))
def test_discriminator(module):
    D = module.Discriminator(img_size=IMG_SIZE, img_depth=IMG_DEPTH)
    assert hasattr(D, "reset_parameters")
    out = D(torch.randn(IMG_SHAPE))
    assert out.shape == (BATCH_SIZE,)
    out.mean().backward()


@pytest.mark.parametrize("module", (dcgan, progressive_gan, stylegan, stylegan2, sagan))
def test_generator(module):
    G = module.Generator(img_size=IMG_SIZE, img_depth=IMG_DEPTH, z_dim=Z_DIM)
    assert hasattr(G, "reset_parameters")
    out = G(torch.randn(NOISE_SHAPE))
    assert out.shape == IMG_SHAPE
    out.mean().backward()


@pytest.mark.parametrize("kernel_size", (3, 4))
@pytest.mark.parametrize("up", (1, 2))
@pytest.mark.parametrize("down", (1, 2))
def test_upfirdn2d_grad_fix(kernel_size, up, down):
    imgs = torch.randn(4, 16, 8, 8, requires_grad=True, dtype=torch.double)
    kernel = torch.randn(kernel_size, kernel_size, dtype=torch.double)
    p1 = (kernel_size - 1) // 2
    p2 = kernel_size - 1 - p1

    # need to square so that 2nd order gradient is non-zero
    loss1 = upfirdn2d(imgs, kernel, up, down, p1, p2, p1, p2).square().sum()
    loss2 = UpFIRDn2d.apply(imgs, kernel, up, down, p1, p2, p1, p2).square().sum()

    # 1st order gradient
    (grad1,) = torch.autograd.grad(loss1, imgs, create_graph=True)
    (grad2,) = torch.autograd.grad(loss2, imgs, create_graph=True)
    assert torch.allclose(grad1, grad2)

    # 2nd order gradient
    (gradgrad1,) = torch.autograd.grad(grad1.sum(), imgs, create_graph=True)
    (gradgrad2,) = torch.autograd.grad(grad2.sum(), imgs, create_graph=True)
    assert torch.allclose(gradgrad1, gradgrad2)


def test_upfirdn2d_avg_pool():
    n, c, h, w = 4, 16, 8, 8
    x = torch.randn(n, c, h, w)
    kernel = Blur.make_kernel(2)

    y1 = F.avg_pool2d(x, 2)
    y2 = upfirdn2d(x, kernel, 1, 2, 0, 1, 0, 1)
    assert torch.allclose(y1, y2)


def test_upfirdn2d_upsample():
    n, c, h, w = 4, 16, 8, 8
    x = torch.randn(n, c, h, w)
    kernel = Blur.make_kernel(2)

    y1 = F.interpolate(x, scale_factor=2)
    y2 = upfirdn2d(x, kernel, 2, 1, 1, 0, 1, 0)
    assert torch.allclose(y1, y2 * 4)
