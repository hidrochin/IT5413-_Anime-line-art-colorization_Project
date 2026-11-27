"""Shared interface every model implements (frozen in week 1).

Keep this signature identical across models: sample.py relies on it to treat A, B and C the same way.
"""

from abc import ABC, abstractmethod

from torch import Tensor


class Colorizer(ABC):
    name: str = "base"

    @abstractmethod
    def colorize(self, sketch: Tensor, reference: Tensor, seed: int, steps: int | None = None) -> Tensor:
        """sketch: (B,1,256,256) in [-1,1]; reference: (B,3,256,256) in [-1,1]
        returns (B,3,256,256) in [-1,1]"""

    @classmethod
    def from_config(cls, cfg: dict) -> "Colorizer":
        """Build from the `model:` section of configs/<model>.yaml (used by scripts/sample.py).
        Default: pass the section as keyword arguments; override if construction is more involved."""
        return cls(**cfg)

    def nfe(self, steps: int | None = None) -> int:
        """Network forward passes per image for `steps` (None = the model's default), for
        results/speed.csv. Override for iterative samplers, counting CFG as 2 passes per step."""
        return 1
