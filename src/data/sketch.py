"""One fixed sketch clean-up, applied to every sketch at load time for all models (same input for all).

The HF mirror mixes two line styles: AnimeDiffusion sketches are binary (pure black lines on white,
plus faint JPEG halos), moeimouto sketches are soft gray pencil-like lines. `binary` maps both to the
AnimeDiffusion style:

- already-binary sketches (< `binary_frac` of pixels mid-gray) are only thresholded at `dark`, which
  removes the JPEG halos without thickening lines;
- gray sketches become black where a pixel is dark (< `dark`) or clearly darker than its 9x9
  neighbourhood (adaptive threshold, C = 6), which keeps faint lines.

Settings picked visually on the local sample (data/sample_preview/sketch_styles.png). `raw` keeps the
files as they are (for the ablation). Select with $ANIME_SKETCH_STYLE (default: binary).
"""

import cv2
import numpy as np

STYLES = ("raw", "binary")


def is_binary(img: np.ndarray, binary_frac: float = 0.01) -> bool:
    mid = (img > 32) & (img < 224)
    return mid.mean() < binary_frac


def binarize(img: np.ndarray, block: int = 9, c: float = 6, dark: int = 128) -> np.ndarray:
    """(H,W) uint8 -> (H,W) uint8 in {0, 255}."""
    black = img < dark
    if not is_binary(img):
        local = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, c)
        black |= local == 0
    return np.where(black, 0, 255).astype(np.uint8)


def normalize_sketch(img: np.ndarray, style: str = "binary") -> np.ndarray:
    if style == "raw":
        return img
    if style == "binary":
        return binarize(img)
    raise ValueError(f"unknown sketch style {style!r}; expected one of {STYLES}")
