import pytest
import torch
import torch.nn.functional as F

import modelling
from modelling import dcgan, progressive_gan, sagan, stylegan, stylegan2
from modelling.nvidia_ops import Blur, UpFIRDn2d, upfirdn2d


IMG_SIZE = 128
IMG_CHANNELS = 3
Z_DIM = 128
BATCH_SIZE = 8
IMG_SHAPE = (BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
NOISE_SHAPE = (BATCH_SIZE, Z_DIM)

MODELS = (dcgan, progressive_gan, stylegan, stylegan2, sagan)


@pytest.mark.parametrize("cls", MODELS)
def test_discriminator(cls):
    D = cls.Discriminator(img_size=IMG_SIZE, img_channels=IMG_CHANNELS)
    assert hasattr(D, "reset_parameters")
    out = D(torch.randn(IMG_SHAPE))
    assert out.shape == (BATCH_SIZE,)
    out.mean().backward()


# @pytest.mark.parametrize("module", MODELS)
# def test_discriminator_compile(module):
#     D = module.Discriminator(img_size=IMG_SIZE, img_channels=IMG_CHANNELS)
#     compiled_D = torch.compile(D)

#     imgs = torch.randn(IMG_SHAPE)
#     torch.testing.assert_close(D(imgs), compiled_D(imgs))


@pytest.mark.parametrize("cls", MODELS)
def test_generator(cls):
    G = cls.Generator(img_size=IMG_SIZE, img_channels=IMG_CHANNELS, z_dim=Z_DIM)
    assert hasattr(G, "reset_parameters")
    out = G(torch.randn(NOISE_SHAPE))
    assert out.shape == IMG_SHAPE
    out.mean().backward()


# @pytest.mark.parametrize("module", MODELS)
# def test_generator_compile(module):
#     G = module.Generator(img_size=IMG_SIZE, img_channels=IMG_CHANNELS, z_dim=Z_DIM)
#     compiled_G = torch.compile(G)

#     noise = torch.randn(NOISE_SHAPE)
#     torch.testing.assert_close(G(noise), compiled_G(noise))


def test_patch_gan():
    m = modelling.PatchGAN(IMG_CHANNELS, IMG_CHANNELS)
    out = m(torch.randn(IMG_SHAPE), torch.randn(IMG_SHAPE))
    assert out.shape == (BATCH_SIZE, 1, IMG_SIZE // 8 - 2, IMG_SIZE // 8 - 2)


@pytest.mark.parametrize("cls", (modelling.UnetGenerator, modelling.ResNetGenerator))
def test_img2img_generator(cls):
    m = cls(IMG_CHANNELS, IMG_CHANNELS)
    out = m(torch.randn(NOISE_SHAPE), torch.randn(IMG_SHAPE))
    assert out.shape == IMG_SHAPE


def test_progressive_gan_discriminator():
    size = 4
    disc = progressive_gan.Discriminator(
        img_size=IMG_SIZE,
        img_channels=IMG_CHANNELS,
        z_dim=Z_DIM,
        progressive_growing=True,
    )

    while True:
        out = disc(torch.randn(BATCH_SIZE, IMG_CHANNELS, size, size))
        assert out.shape == (BATCH_SIZE,)
        size *= 2
        if size > IMG_SIZE:
            break
        disc.grow()


def test_progressive_gan_generator():
    size = 4
    gen = progressive_gan.Generator(
        img_size=IMG_SIZE,
        img_channels=IMG_CHANNELS,
        z_dim=Z_DIM,
        progressive_growing=True,
    )

    while True:
        out = gen(torch.randn(NOISE_SHAPE))
        assert out.shape[-2:] == (size, size)
        size *= 2
        if size > IMG_SIZE:
            break
        gen.grow()


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
    torch.testing.assert_close(grad1, grad2)

    # 2nd order gradient
    (gradgrad1,) = torch.autograd.grad(grad1.sum(), imgs, create_graph=True)
    (gradgrad2,) = torch.autograd.grad(grad2.sum(), imgs, create_graph=True)
    torch.testing.assert_close(gradgrad1, gradgrad2)


def test_upfirdn2d_avg_pool():
    n, c, h, w = 4, 16, 8, 8
    x = torch.randn(n, c, h, w)
    kernel = Blur.make_kernel(2)

    y1 = F.avg_pool2d(x, 2)
    y2 = upfirdn2d(x, kernel, 1, 2, 0, 1, 0, 1)
    torch.testing.assert_close(y1, y2)


def test_upfirdn2d_upsample():
    n, c, h, w = 4, 16, 8, 8
    x = torch.randn(n, c, h, w)
    kernel = Blur.make_kernel(2)

    y1 = F.interpolate(x, scale_factor=2)
    y2 = upfirdn2d(x, kernel, 2, 1, 1, 0, 1, 0)
    torch.testing.assert_close(y1, y2 * 4)
