from functools import partial

from torch import nn

from .base import _Act, _Norm, conv_norm_act, make_layers


class Discriminator(nn.Sequential):
    def __init__(self, norm: _Norm = nn.BatchNorm2d, act: _Act = partial(nn.LeakyReLU, 0.2, True)):
        super().__init__()
        kwargs = dict(kernel_size=4, stride=2, padding=1, norm=norm, act=act)
        layer_configs = [[64], [128], [256], [512]]
        self.convs = nn.Sequential(*make_layers(3, layer_configs, **kwargs))
        self.out_conv = nn.Conv2d(512, 1, 4)

        self.apply(init_weights)


class Generator(nn.Module):
    def __init__(self, z_dim: int, norm: _Norm = nn.BatchNorm2d, act: _Act = nn.ReLU):
        super().__init__()
        kwargs = dict(conv=nn.ConvTranspose2d, norm=norm, act=act)
        self.in_conv = conv_norm_act(z_dim, 512, 4, **kwargs)
        layer_configs = [[256], [128], [64], [32]]
        self.ups = nn.Sequential(
            *make_layers(512, layer_configs, kernel_size=4, stride=2, padding=1, **kwargs),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh(),
        )

        self.apply(init_weights)
    
    def forward(self, x):
        x = self.in_conv(x[:, :, None, None])
        return self.ups(x)


# https://github.com/soumith/dcgan.torch/blob/master/main.lua#L42
# https://github.com/martinarjovsky/WassersteinGAN/blob/master/main.py#L108
def init_weights(module: nn.Module):
    if isinstance(module, nn.modules.conv._ConvNd):
        nn.init.normal_(module.weight, 0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.modules.batchnorm._BatchNorm):
        nn.init.normal_(module.weight, 1, 0.02)
        nn.init.zeros_(module.bias)
