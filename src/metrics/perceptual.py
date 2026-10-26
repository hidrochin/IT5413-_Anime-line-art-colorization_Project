"""LPIPS to ground truth, and diversity as mean pairwise LPIPS between seeds."""

from itertools import combinations

import torch
from torch import Tensor


class LPIPS:
    """Callable LPIPS distance on (B,3,H,W) images in [0,1] -> (B,)."""

    def __init__(self, net: str = "alex", device: str | torch.device = "cpu"):
        import lpips

        self.device = torch.device(device)
        self.model = lpips.LPIPS(net=net, verbose=False).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        a = a.to(self.device, torch.float32) * 2 - 1
        b = b.to(self.device, torch.float32) * 2 - 1
        return self.model(a, b).flatten().cpu()


def pairwise_diversity(samples: Tensor, lpips_fn) -> float:
    """Mean LPIPS over all pairs of K samples (K,3,H,W) generated for one sketch/reference."""
    k = samples.shape[0]
    if k < 2:
        return float("nan")
    i, j = zip(*combinations(range(k), 2))
    return lpips_fn(samples[list(i)], samples[list(j)]).mean().item()
