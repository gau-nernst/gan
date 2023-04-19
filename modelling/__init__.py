from .cgan import CGANDiscriminator, CGANGenerator
from .cyclegan import ResNetGenerator
from .dcgan import DCGANDiscriminator, DCGANGenerator
from .pix2pix import PatchGAN, UnetGenerator
from .progressive_gan import ProgressiveGANDiscriminator, ProgressiveGANGenerator
from .sagan import SAGANDiscriminator, SAGANGenerator
from .stylegan import StyleGANDiscriminator, StyleGANGenerator
from .stylegan2 import StyleGAN2Discriminator, StyleGAN2Generator


def get_generator_cls(name: str):
    return dict(
        dcgan=DCGANGenerator,
        progressive_gan=ProgressiveGANGenerator,
        stylegan=StyleGANGenerator,
        stylegan2=StyleGAN2Generator,
        sagan=SAGANGenerator,
        cgan=CGANGenerator,
        pix2pix=UnetGenerator,
        cyclegan=ResNetGenerator,
    )[name]


def get_discriminator_cls(name: str):
    return dict(
        dcgan=DCGANDiscriminator,
        progressive_gan=ProgressiveGANDiscriminator,
        stylegan=StyleGANDiscriminator,
        stylegan2=StyleGAN2Discriminator,
        sagan=SAGANDiscriminator,
        cgan=CGANDiscriminator,
        pix2pix=PatchGAN,
        cyclegan=PatchGAN,
    )[name]
