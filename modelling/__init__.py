from .cgan import CGANDiscriminator, CGANGenerator
from .cyclegan import ResNetGenerator
from .dcgan import DcGanDiscriminator, DcGanGenerator
from .esrgan import ESRGANGenerator
from .pix2pix import PatchGAN, UnetGenerator
from .progressive_gan import ProgressiveGANDiscriminator, ProgressiveGANGenerator
from .sagan import SaGanDiscriminator, SaGanGenerator
from .srgan import SRGANDiscriminator, SRResNet
from .stylegan import StyleGANDiscriminator, StyleGANGenerator
from .stylegan2 import StyleGAN2Discriminator, StyleGAN2Generator


def get_generator_cls(name: str):
    return dict(
        dcgan=DcGanGenerator,
        progressive_gan=ProgressiveGANGenerator,
        stylegan=StyleGANGenerator,
        stylegan2=StyleGAN2Generator,
        sagan=SaGanGenerator,
        cgan=CGANGenerator,
        pix2pix=UnetGenerator,
        cyclegan=ResNetGenerator,
        srgan=SRResNet,
        esrgan=ESRGANGenerator,
    )[name]


def get_discriminator_cls(name: str):
    return dict(
        dcgan=DcGanDiscriminator,
        progressive_gan=ProgressiveGANDiscriminator,
        stylegan=StyleGANDiscriminator,
        stylegan2=StyleGAN2Discriminator,
        sagan=SaGanDiscriminator,
        cgan=CGANDiscriminator,
        pix2pix=PatchGAN,
        cyclegan=PatchGAN,
        srgan=SRGANDiscriminator,
    )[name]
