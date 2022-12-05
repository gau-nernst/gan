from typing import Literal

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch import Tensor, nn


class GANSystem(pl.LightningModule):
    def __init__(
        self,
        discriminator: nn.Module,
        generator: nn.Module,
        z_dim: int,
        method: Literal["gan", "wgan", "wgan-gp"] = "gan",
        label_smoothing: float = 0.0,
        clip: float = 0.01,
        lamb: float = 10.0,
        train_g_interval: int = 1,
        log_img_interval: int = 1000,
    ):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.save_hyperparameters(ignore=["discriminator", "generator"])

        self.automatic_optimization = False

    def generate(self, bsize: int):
        z_noise = torch.randn(bsize, self.hparams["z_dim"], device=self.device)
        return self.generator(z_noise)

    def _optim_step(self, loss: torch.Tensor, optim: LightningOptimizer):
        optim.zero_grad()
        self.manual_backward(loss)
        optim.step()

    def configure_optimizers(self):
        optim_d = torch.optim.Adam(self.discriminator.parameters(), 1e-4, betas=(0.5, 0.999))
        optim_g = torch.optim.Adam(self.generator.parameters(), 1e-4, betas=(0.5, 0.999))
        return optim_d, optim_g

    def configure_callbacks(self):
        fixed_noise = torch.randn((32, self.hparams["z_dim"]))
        return [ImageLoggingCallback(self.hparams["log_img_interval"], fixed_noise)]

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        x_reals, _ = batch
        optim_d, optim_g = self.optimizers()
        bsize = x_reals.size(0)
        method = self.hparams["method"]

        # train D
        with torch.no_grad():
            x_fakes = self.generate(bsize)
        d_reals, d_fakes = map(self.discriminator, (x_reals, x_fakes))

        if method == "gan":
            loss_d_real = -F.logsigmoid(d_reals).mean() * (1.0 - self.hparams["label_smoothing"])
            loss_d_fake = -F.logsigmoid(-d_fakes).mean()
            loss_d = loss_d_real + loss_d_fake

        elif method in ("wgan", "wgan-gp"):
            loss_d = d_fakes.mean() - d_reals.mean()

            if method == "wgan-gp":
                alpha = torch.rand(bsize, 1, 1, 1, device=self.device)
                x_inters = (x_reals * alpha + x_fakes * (1 - alpha)).requires_grad_()
                d_inters = self.discriminator(x_inters)

                d_grad = torch.autograd.grad(d_inters, x_inters, torch.ones_like(d_inters), create_graph=True)[0]
                d_grad_norm = torch.linalg.vector_norm(d_grad, dim=(1, 2, 3))
                loss_d = loss_d + self.hparams["lamb"] * ((d_grad_norm - 1) ** 2).mean()

        else:
            raise ValueError

        self._optim_step(loss_d, optim_d)
        self.log("loss_d", loss_d)

        if method == "wgan":
            clip = self.hparams["clip"]
            with torch.no_grad():
                for param in self.discriminator.parameters():
                    param.clip_(-clip, clip)

        # train G
        if (batch_idx + 1) % self.hparams["train_g_interval"] == 0:
            d_fakes = self.discriminator(self.generate(bsize))

            if method == "gan":
                loss_g = -F.logsigmoid(d_fakes).mean()

            elif method in ("wgan", "wgan-gp"):
                loss_g = -self.discriminator(self.generate(bsize)).mean()

            else:
                raise ValueError

            self._optim_step(loss_g, optim_g)
            self.log("loss_g", loss_g)


class ImageLoggingCallback(pl.Callback):
    def __init__(self, log_interval: int, fixed_noise: Tensor):
        super().__init__()
        self.log_interval = log_interval
        self.fixed_noise = fixed_noise

    @torch.no_grad()
    def _log_images(self, pl_module: GANSystem):
        images = pl_module.generator(self.fixed_noise)
        pl_module.logger.experiment.add_images("generated", images, pl_module.global_step)

    def on_train_start(self, trainer: pl.Trainer, pl_module: GANSystem) -> None:
        if trainer.is_global_zero:
            self.fixed_noise = self.fixed_noise.to(pl_module.device)
            self._log_images(pl_module)

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: GANSystem, *args) -> None:
        if trainer.is_global_zero and pl_module.global_step % self.log_interval == 0:
            self._log_images(pl_module)
