# DCGAN - https://arxiv.org/abs/1701.07875
# Discriminator will downsample until the feature map is 4x4, then flatten and matmul
# Generator will matmul, reshape to 4x4, then upsample until the desired image size is obtained
#
# Code references:
# https://github.com/soumith/dcgan.torch/
# https://github.com/martinarjovsky/WassersteinGAN/

import torch
from torch import Tensor, nn


class CGANDiscriminator(nn.Module):
    def __init__(self, D: nn.Module, y_encoder: nn.Module):
        super().__init__()
        self.D = D
        self.y_encoder = y_encoder
        self.y_encoder.apply(init_weights)

    def forward(self, imgs: Tensor, ys: Tensor):
        n, _, h, w = imgs.shape
        y_embs = self.y_encoder(ys).view(n, -1, 1, 1).expand(-1, -1, h, w)
        imgs = torch.cat([imgs, y_embs], dim=1)
        return self.D(imgs)


class CGANGenerator(nn.Module):
    def __init__(self, G: nn.Module, y_encoder: nn.Module):
        super().__init__()
        self.G = G
        self.y_encoder = y_encoder
        self.y_encoder.apply(init_weights)

    def forward(self, z_embs: Tensor, ys: Tensor):
        y_embs = self.y_encoder(ys)
        z_embs = torch.cat([z_embs, y_embs], dim=1)
        return self.G(z_embs)


def init_weights(module: nn.Module):
    if hasattr(module, "weight"):
        nn.init.normal_(module.weight, 0, 0.02)
    if hasattr(module, "bias"):
        nn.init.zeros_(module.bias)
