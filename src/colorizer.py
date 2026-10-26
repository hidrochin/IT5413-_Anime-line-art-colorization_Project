"""Shared interface every model implements (frozen in week 1; owner: Thành).

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
