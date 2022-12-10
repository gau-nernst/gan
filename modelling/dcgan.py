from functools import partial
from typing import Optional, List

import torch
from torch import nn, Tensor

from .base import _Act, _Norm, conv_norm_act


class Discriminator(nn.Module):
    def __init__(
        self,
        img_size: int = 64,
        img_depth: int = 3,
        depth_list: Optional[List[int]] = None,
        norm: _Norm = nn.BatchNorm2d,
        act: _Act = partial(nn.LeakyReLU, 0.2, True),
        c_dim: int = 0,
    ):
        super().__init__()
        kwargs = dict(kernel_size=4, stride=2, padding=1, norm=norm, act=act)
        depth_list = depth_list or [64, 128, 256, 512]
        last_feat_map_size = img_size // 2 ** len(depth_list)
        img_depth += c_dim

        self.convs = nn.Sequential()
        for depth in depth_list:
            self.convs.append(conv_norm_act(img_depth, depth, **kwargs))
            img_depth = depth
        self.convs.append(nn.Conv2d(img_depth, 1, last_feat_map_size))

        self.apply(init_weights)

    def forward(self, imgs: Tensor, c_embs: Optional[Tensor] = None):
        if c_embs is not None:
            img_h, img_w = imgs.shape[2:]
            c_embs = c_embs[:, :, None, None].expand(-1, -1, img_h, img_w)
            imgs = torch.cat([imgs, c_embs], dim=1)
        return self.convs(imgs)


class Generator(nn.Module):
    def __init__(
        self,
        img_size: int = 64,
        img_depth: int = 3,
        z_dim: int = 128,
        depth_list: Optional[List[int]] = None,
        norm: _Norm = nn.BatchNorm2d,
        act: _Act = partial(nn.ReLU, True),
        c_dim: int = 0,
    ):
        super().__init__()
        kwargs = dict(conv=nn.ConvTranspose2d, norm=norm, act=act)
        depth_list = depth_list or [512, 256, 128, 64]
        first_feat_map_size = img_size // 2 ** len(depth_list)
        z_dim += c_dim

        self.convs = nn.Sequential()
        self.convs.append(conv_norm_act(z_dim, depth_list[0], first_feat_map_size, **kwargs))
        in_depth = depth_list[0]
        kwargs.update(kernel_size=4, stride=2, padding=1)
        for depth in depth_list[1:]:
            self.convs.append(conv_norm_act(in_depth, depth, **kwargs))
            in_depth = depth
        kwargs.update(norm=None, act=nn.Tanh)
        self.convs.append(conv_norm_act(in_depth, img_depth, **kwargs))

        self.apply(init_weights)

    def forward(self, z_embs: Tensor, c_embs: Optional[Tensor] = None):
        if c_embs is not None:
            z_embs = torch.cat([z_embs, c_embs], dim=1)
        return self.convs(z_embs[:, :, None, None])


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
