from .cgan import CGANDiscriminator, CGANGenerator
from .cyclegan import ResNetGenerator
from .dcgan import DcGanDiscriminator, DcGanGenerator
from .esrgan import ESRGANGenerator
from .pix2pix import PatchGan, UnetGenerator
from .progressive_gan import ProgressiveGanDiscriminator, ProgressiveGanGenerator
from .sagan import SaGanDiscriminator, SaGanGenerator
from .srgan import SRGANDiscriminator, SRResNet
from .stargan import StarGanDiscriminator, StarGanGenerator
from .starganv2 import StarGanv2Generator
from .stylegan import StyleGanGenerator
from .stylegan2 import StyleGan2Generator


def build_generator(name: str, *args, **kwargs):
    return dict(
        dcgan=DcGanGenerator,
        progressive_gan=ProgressiveGanGenerator,
        stylegan=StyleGanGenerator,
        stylegan2=StyleGan2Generator,
        sagan=SaGanGenerator,
        cgan=CGANGenerator,
        pix2pix=UnetGenerator,
        cyclegan=ResNetGenerator,
        srgan=SRResNet,
        esrgan=ESRGANGenerator,
        stargan=StarGanGenerator,
        starganv2=StarGanv2Generator,
    )[name](*args, **kwargs)


def build_discriminator(name: str, *args, **kwargs):
    return dict(
        dcgan=DcGanDiscriminator,
        progressive_gan=ProgressiveGanDiscriminator,
        stylegan=ProgressiveGanDiscriminator,
        stylegan2=ProgressiveGanDiscriminator,
        sagan=SaGanDiscriminator,
        cgan=CGANDiscriminator,
        pix2pix=PatchGan,
        cyclegan=PatchGan,
        srgan=SRGANDiscriminator,
        stargan=StarGanDiscriminator,
        # starganv2=StarGanv2Discriminator,
    )[name](*args, **kwargs)
