"""Paired fidelity metrics (self-reference mode only). Inputs: (B,3,H,W) float in [0,1]."""

import torch
from torch import Tensor
from torchmetrics.functional.image import (
    multiscale_structural_similarity_index_measure,
    structural_similarity_index_measure,
)


def psnr(pred: Tensor, target: Tensor) -> Tensor:
    """Per-image PSNR in dB, (B,). Identical images are capped at 100 dB."""
    mse = (pred.float() - target.float()).pow(2).flatten(1).mean(1)
    return 10 * torch.log10(1.0 / mse.clamp_min(1e-10))


def ssim(pred: Tensor, target: Tensor) -> Tensor:
    return structural_similarity_index_measure(pred.float(), target.float(), data_range=1.0, reduction="none")


def ms_ssim(pred: Tensor, target: Tensor) -> Tensor:
    """Needs H, W > 160 with the default 5 scales; fine at 256."""
    return multiscale_structural_similarity_index_measure(
        pred.float(), target.float(), data_range=1.0, reduction="none"
    )
