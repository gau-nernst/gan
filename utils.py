import argparse
import copy
import inspect
import json

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


def make_parser(fn):
    parser = argparse.ArgumentParser()

    for k, v in inspect.signature(fn).parameters.items():
        if v.annotation in (str, int, float):
            parser.add_argument(f"--{k}", type=v.annotation, default=v.default)
        elif v.annotation is bool:
            parser.add_argument(f"--{k}", action="store_true")
        elif v.annotation is dict:
            parser.add_argument(f"--{k}", type=json.loads, default=dict())
        else:
            raise RuntimeError(f"Unsupported type {v.annotation} for argument {k}")

    return parser


def normalize_img(x: Tensor) -> Tensor:
    return (x / 255 - 0.5) / 0.5


def unnormalize_img(x: Tensor) -> Tensor:
    return ((x * 0.5 + 0.5) * 255).round().clip(0, 255).to(torch.uint8)


def apply_spectral_norm(m: nn.Module):
    if isinstance(m, (nn.Linear, nn.modules.conv._ConvNd, nn.Embedding)):
        nn.utils.parametrizations.spectral_norm(m)


def prepare_model(
    model: nn.Module, *, spectral_norm: bool = False, channels_last: bool = False, compile: bool = False
) -> None:
    if spectral_norm:
        model.apply(apply_spectral_norm)
    if channels_last:
        model.to(memory_format=torch.channels_last)
    if compile:
        model.compile()


def _tensor_to(x, *, device=None, channels_last: bool = False):
    if isinstance(x, Tensor):
        if channels_last and x.ndim == 4:
            return x.to(device=device, memory_format=torch.channels_last)
        else:
            return x.to(device=device)
    elif isinstance(x, (tuple, list)):
        return x.__class__(_tensor_to(item, device=device, channels_last=channels_last) for item in x)
    else:
        raise ValueError(f"Unuspported type {type(x)}")


def cycle(iterator, *, device=None, channels_last: bool = False):
    while True:
        for x in iterator:
            yield _tensor_to(x, device=device, channels_last=channels_last)


def prepare_train_dloader(ds: Dataset, batch_size: int, *, device=None, channels_last: bool = False):
    dloader = DataLoader(ds, batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    return cycle(dloader, device=device, channels_last=channels_last)


# reference: https://github.com/lucidrains/ema-pytorch
class EMA(nn.Module):
    def __init__(self, model: nn.Module, beta: float = 0.999, warmup: int = 100) -> None:
        super().__init__()
        self.ema_model = copy.deepcopy(model)
        self.model = [model]
        self.beta = beta
        self.warmup = warmup
        self.register_buffer("counter", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def step(self) -> None:
        self.counter += 1
        if self.counter.item() < self.warmup:
            return

        if self.counter.item() == self.warmup:
            for p, ema_p in zip(self.model[0].parameters(), self.ema_model.parameters()):
                ema_p.copy_(p)
            return

        for p, ema_p in zip(self.model[0].parameters(), self.ema_model.parameters()):
            ema_p.lerp_(p, 1 - self.beta)

    def forward(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)
