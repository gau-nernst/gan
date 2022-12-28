from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch import Tensor, nn


def apply_spectral_normalization(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, (nn.modules.conv._ConvNd, nn.Linear)):
            setattr(module, name, nn.utils.parametrizations.spectral_norm(child))
        else:
            apply_spectral_normalization(child)


class GANSystem(pl.LightningModule):
    def __init__(
        self,
        discriminator: nn.Module,
        generator: nn.Module,
        conditional: bool = False,
        z_dim: int = 128,
        method: str = "gan",
        spectral_norm_d: bool = False,
        spectral_norm_g: bool = False,
        label_smoothing: float = 0.0,
        clip: float = 0.01,
        lamb: float = 10.0,
        train_g_interval: int = 1,
        optimizer: str = "Adam",
        lr: float = 2e-4,
        lr_d: Optional[float] = None,
        lr_g: Optional[float] = None,
        weight_decay: float = 0,
        beta1: float = 0.5,
        beta2: float = 0.999,
        **kwargs,
    ):
        assert method in ("gan", "wgan", "wgan-gp", "hinge")
        assert optimizer in ("SGD", "Adam", "AdamW", "RMSprop")
        assert clip > 0
        super().__init__()
        # TODO: fix model checkpoint when spectral norm is applied
        if spectral_norm_d:
            apply_spectral_normalization(discriminator)
        if spectral_norm_g:
            apply_spectral_normalization(generator)
        self.discriminator = discriminator
        self.generator = generator
        lr_d = lr_d or lr
        lr_g = lr_g or lr
        self.save_hyperparameters(ignore=["discriminator", "generator", "condition_encoder"])

        self.automatic_optimization = False

    def generate(self, bsize: int, ys: Optional[Tensor] = None) -> Tensor:
        z_noise = torch.randn(bsize, self.hparams["z_dim"], device=self.device)
        return self.generator(z_noise) if ys is None else self.generator(z_noise, ys)

    def _optim_step(self, loss: Tensor, optim: LightningOptimizer):
        optim.zero_grad()
        self.manual_backward(loss)
        optim.step()

    def configure_optimizers(self):
        optim_cls = getattr(torch.optim, self.hparams["optimizer"])
        kwargs = dict(weight_decay=self.hparams["weight_decay"])
        if self.hparams["optimizer"] in ("Adam", "AdamW"):
            kwargs.update(betas=(self.hparams["beta1"], self.hparams["beta2"]))

        optim_d = optim_cls(self.discriminator.parameters(), lr=self.hparams["lr_d"], **kwargs)

        if hasattr(self.generator, "mapping_network"):  # stylegan
            group2 = self.generator.mapping_network.parameters()
            group1 = [p for p in self.generator.parameters() if p not in group2]
            param_groups = [
                dict(params=group1),
                dict(params=group2, lr=self.hparams["lr_g"] / 100),
            ]
            optim_g = optim_cls(param_groups, **kwargs)
        else:
            param_groups = [dict(params=list(self.generator.parameters()))]
        optim_g = optim_cls(param_groups, lr=self.hparams["lr_g"], **kwargs)

        return optim_d, optim_g

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        x_reals, ys = batch
        optim_d, optim_g = self.optimizers()
        bsize = x_reals.size(0)
        method = self.hparams["method"]
        conditional = self.hparams["conditional"]

        # train D
        if conditional:
            with torch.no_grad():
                x_fakes = self.generate(bsize, ys)
            d_reals = self.discriminator(x_reals, ys)
            d_fakes = self.discriminator(x_fakes, ys)
        else:
            with torch.no_grad():
                x_fakes = self.generate(bsize)
            d_reals = self.discriminator(x_reals)
            d_fakes = self.discriminator(x_fakes)

        if method == "gan":
            loss_d_real = -F.logsigmoid(d_reals).mean() * (1.0 - self.hparams["label_smoothing"])
            loss_d_fake = -F.logsigmoid(-d_fakes).mean()
            loss_d = loss_d_real + loss_d_fake

        elif method in ("wgan", "wgan-gp"):
            loss_d = d_fakes.mean() - d_reals.mean()

            if method == "wgan-gp":
                alpha = torch.rand(bsize, 1, 1, 1, device=self.device)
                x_inters = (x_reals * alpha + x_fakes * (1 - alpha)).requires_grad_()
                d_inters = self.discriminator(x_inters, ys) if conditional else self.discriminator(x_inters)

                (d_grad,) = torch.autograd.grad(d_inters, x_inters, torch.ones_like(d_inters), create_graph=True)
                d_grad_norm = torch.linalg.vector_norm(d_grad, dim=(1, 2, 3))
                loss_d = loss_d + self.hparams["lamb"] * ((d_grad_norm - 1) ** 2).mean()

        elif method == "hinge":
            loss_d = F.relu(1 - d_reals).mean() + F.relu(1 + d_fakes).mean()

        else:
            raise ValueError

        self._optim_step(loss_d, optim_d)
        self.log("loss_d", loss_d)

        # Algorithm 1 in paper clip weights after optimizer step, but GitHub code clip before optimizer step
        # it doesn't seem to matter much in practice
        if method == "wgan":
            clip = self.hparams["clip"]
            with torch.no_grad():
                for param in self.discriminator.parameters():
                    param.clip_(-clip, clip)

        # train G
        if (batch_idx + 1) % self.hparams["train_g_interval"] == 0:
            self.discriminator.requires_grad_(False)

            if conditional:
                x_fakes = self.generate(bsize, ys)
                d_fakes = self.discriminator(x_fakes, ys)
            else:
                x_fakes = self.generate(bsize)
                d_fakes = self.discriminator(x_fakes)

            if method == "gan":
                loss_g = -F.logsigmoid(d_fakes).mean()

            elif method in ("wgan", "wgan-gp", "hinge"):
                loss_g = -d_fakes.mean()

            else:
                raise ValueError

            self._optim_step(loss_g, optim_g)
            self.log("loss_g", loss_g)

            self.discriminator.requires_grad_(True)

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        pass


class ImageLoggingCallback(pl.Callback):
    def __init__(self, log_interval: int, fixed_noise: Tensor, fixed_y: Optional[Tensor] = None):
        super().__init__()
        self.log_interval = log_interval
        self.fixed_noise = fixed_noise
        self.fixed_y = fixed_y

    @torch.no_grad()
    def _log_images(self, pl_module: GANSystem):
        generator = pl_module.generator
        logger = pl_module.logger
        global_step = pl_module.global_step

        images = generator(self.fixed_noise, self.fixed_y).mul_(0.5).add_(0.5)
        if isinstance(logger, TensorBoardLogger):
            logger.experiment.add_images("generated", images, global_step)
        elif isinstance(logger, WandbLogger):
            logger.log_image("generated", images)

    def on_train_start(self, trainer: pl.Trainer, pl_module: GANSystem) -> None:
        if trainer.is_global_zero:
            self.fixed_noise = self.fixed_noise.to(pl_module.device)
            if self.fixed_y is not None:
                self.fixed_y = self.fixed_y.to(pl_module.device)
            self._log_images(pl_module)

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: GANSystem, *args) -> None:
        if trainer.is_global_zero and pl_module.global_step % self.log_interval == 0:
            self._log_images(pl_module)
