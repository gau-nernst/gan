from .cgan import CGANDiscriminator, CGANGenerator
from .cyclegan import ResNetGenerator
from .dcgan import DcGanDiscriminator, DcGanGenerator
from .esrgan import ESRGANGenerator
from .pix2pix import PatchGAN, UnetGenerator
from .progressive_gan import ProgressiveGanDiscriminator, ProgressiveGanGenerator
from .sagan import SaGanDiscriminator, SaGanGenerator
from .srgan import SRGANDiscriminator, SRResNet
from .stylegan import StyleGanGenerator
from .stylegan2 import StyleGAN2Generator


def build_generator(name: str, *args, **kwargs):
    return dict(
        dcgan=DcGanGenerator,
        progressive_gan=ProgressiveGanGenerator,
        stylegan=StyleGanGenerator,
        stylegan2=StyleGAN2Generator,
        sagan=SaGanGenerator,
        cgan=CGANGenerator,
        pix2pix=UnetGenerator,
        cyclegan=ResNetGenerator,
        srgan=SRResNet,
        esrgan=ESRGANGenerator,
    )[name](*args, **kwargs)


def build_discriminator(name: str, *args, **kwargs):
    return dict(
        dcgan=DcGanDiscriminator,
        progressive_gan=ProgressiveGanDiscriminator,
        stylegan=ProgressiveGanDiscriminator,
        stylegan2=ProgressiveGanDiscriminator,
        sagan=SaGanDiscriminator,
        cgan=CGANDiscriminator,
        pix2pix=PatchGAN,
        cyclegan=PatchGAN,
        srgan=SRGANDiscriminator,
    )[name](*args, **kwargs)
