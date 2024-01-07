import copy

import torch
from torch import nn


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
