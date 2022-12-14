# DCGAN - https://arxiv.org/abs/1701.07875
# Discriminator will downsample until the feature map is 4x4, then flatten and matmul
# Generator will matmul, reshape to 4x4, then upsample until the desired image size is obtained
#
# Code references:
# https://github.com/soumith/dcgan.torch/
# https://github.com/martinarjovsky/WassersteinGAN/

import torch
from torch import Tensor, nn


class Discriminator(nn.Module):
    def __init__(self, D: nn.Module, y_encoder: nn.Module):
        super().__init__()
        self.D = D
        self.y_encoder = y_encoder

    def forward(self, imgs: Tensor, ys: Tensor):
        n, c, h, w = imgs.shape
        y_embs = self.y_encoder(ys).view(n, -1, 1, 1).expand(-1, -1, h, w)
        imgs = torch.cat([imgs, y_embs], dim=1)
        return self.D(imgs)


class Generator(nn.Module):
    def __init__(self, G: nn.Module, y_encoder: nn.Module):
        super().__init__()
        self.G = G
        self.y_encoder = y_encoder

    def forward(self, z_embs: Tensor, ys: Tensor):
        y_embs = self.y_encoder(ys)
        z_embs = torch.cat([z_embs, y_embs], dim=1)
        return self.G(z_embs)
