import pytest
import torch
import torch.nn.functional as F

from modelling import dcgan, progressive_gan, sagan, stylegan, stylegan2
from modelling.nvidia_ops import Blur, upfirdn2d

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
