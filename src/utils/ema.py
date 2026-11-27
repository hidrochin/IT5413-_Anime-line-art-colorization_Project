"""Exponential moving average of model weights (Model B uses decay 0.9999; sample from the EMA copy)."""

import copy

import torch
from torch import nn

from src.utils.checkpoint import unwrap


class EMA:
    """Keeps a frozen copy of `model` whose weights track `decay * ema + (1 - decay) * model`.

        ema = EMA(model, decay=0.9999)
        ...optimizer.step(); ema.update(model)
        sampler_model = ema.model
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, warmup: bool = True):
        self.model = copy.deepcopy(unwrap(model)).eval().requires_grad_(False)
        self.decay = decay
        self.warmup = warmup  # use a smaller decay early on, so the EMA isn't stuck near the init
        self.num_updates = 0

    def current_decay(self) -> float:
        if not self.warmup:
            return self.decay
        return min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        d = self.current_decay()
        src = unwrap(model).state_dict()
        for name, ema_v in self.model.state_dict().items():
            v = src[name].detach()
            if ema_v.dtype.is_floating_point:
                ema_v.lerp_(v.to(ema_v.dtype), 1 - d)
            else:
                ema_v.copy_(v)  # integer buffers (e.g. BatchNorm counters)

    def state_dict(self) -> dict:
        return {"model": self.model.state_dict(), "decay": self.decay, "num_updates": self.num_updates}

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.decay = state["decay"]
        self.num_updates = state["num_updates"]
