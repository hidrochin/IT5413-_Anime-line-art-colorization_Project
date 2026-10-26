"""Reference agreement (cross-reference mode): Lab color-histogram distance and CLIP similarity.

The face/hair mask is still an open question in CLAUDE.md. The default here treats the image's
border color as background and keeps pixels that differ from it by more than `mask_delta_e`;
the same rule is applied to output and reference so the comparison stays symmetric.
"""

import cv2
import numpy as np
import torch

LAB_RANGES = ((0.0, 100.0), (-110.0, 110.0), (-110.0, 110.0))


def rgb_to_lab(rgb_uint8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb_uint8.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def foreground_mask(lab: np.ndarray, delta_e: float = 10.0, border: int = 4, min_fraction: float = 0.05) -> np.ndarray:
    """Pixels whose color differs from the median border color; falls back to all pixels."""
    edge = np.concatenate(
        [lab[:border].reshape(-1, 3), lab[-border:].reshape(-1, 3), lab[:, :border].reshape(-1, 3), lab[:, -border:].reshape(-1, 3)]
    )
    background = np.median(edge, axis=0)
    mask = np.linalg.norm(lab - background, axis=-1) > delta_e
    if mask.mean() < min_fraction:
        return np.ones(lab.shape[:2], dtype=bool)
    return mask


def lab_histogram(lab: np.ndarray, mask: np.ndarray | None = None, bins: tuple[int, int, int] = (8, 16, 16)) -> np.ndarray:
    pixels = lab.reshape(-1, 3) if mask is None else lab[mask]
    hist, _ = np.histogramdd(pixels, bins=bins, range=LAB_RANGES)
    return hist.ravel() / max(hist.sum(), 1.0)


def color_distance(pred_rgb: np.ndarray, ref_rgb: np.ndarray, use_mask: bool = True) -> float:
    """Hellinger distance in [0,1] between masked Lab histograms; 0 = identical color distribution."""
    lab_p, lab_r = rgb_to_lab(pred_rgb), rgb_to_lab(ref_rgb)
    mask_p = foreground_mask(lab_p) if use_mask else None
    mask_r = foreground_mask(lab_r) if use_mask else None
    p, q = lab_histogram(lab_p, mask_p), lab_histogram(lab_r, mask_r)
    return float(np.sqrt(max(0.0, 1.0 - np.sqrt(p * q).sum())))


class CLIPSimilarity:
    """Cosine similarity of CLIP image embeddings between outputs and references."""

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", device: str | torch.device = "cpu"):
        from transformers import CLIPImageProcessor, CLIPModel

        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPImageProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def embed(self, images_uint8: list[np.ndarray]) -> torch.Tensor:
        pixels = self.processor(images=images_uint8, return_tensors="pt").pixel_values.to(self.device)
        feats = self.model.get_image_features(pixel_values=pixels)
        if not isinstance(feats, torch.Tensor):  # transformers >= 5 returns an output object
            feats = feats.pooler_output
        return torch.nn.functional.normalize(feats.float(), dim=-1).cpu()

    def __call__(self, preds_uint8: list[np.ndarray], refs_uint8: list[np.ndarray]) -> torch.Tensor:
        return (self.embed(preds_uint8) * self.embed(refs_uint8)).sum(-1)
