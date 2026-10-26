"""Augmented self-reference: a random thin-plate-spline warp + horizontal flip + crop of the ground truth.

Colors stay correct but positions no longer line up with the sketch, so a model must learn
correspondence instead of copying pixels (SCFT, Lee et al. 2020).
"""

import cv2
import numpy as np


def _tps_kernel(r2: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        k = r2 * np.log(r2)
    return np.nan_to_num(k)


def tps_grid(src: np.ndarray, dst: np.ndarray, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Sampling maps (map_x, map_y) for cv2.remap so that output point dst[i] samples input point src[i].
    Points are (N,2) arrays of (x, y) pixel coordinates."""
    n = len(dst)
    k = _tps_kernel(((dst[:, None] - dst[None]) ** 2).sum(-1))
    p = np.hstack([np.ones((n, 1)), dst])
    a = np.zeros((n + 3, n + 3))
    a[:n, :n], a[:n, n:], a[n:, :n] = k, p, p.T
    b = np.zeros((n + 3, 2))
    b[:n] = src
    coef = np.linalg.solve(a + 1e-8 * np.eye(n + 3), b)

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    grid = np.stack([xs.ravel(), ys.ravel()], 1)
    u = _tps_kernel(((grid[:, None] - dst[None]) ** 2).sum(-1))
    mapped = u @ coef[:n] + np.hstack([np.ones((len(grid), 1)), grid]) @ coef[n:]
    return mapped[:, 0].reshape(h, w).astype(np.float32), mapped[:, 1].reshape(h, w).astype(np.float32)


def random_tps(img: np.ndarray, rng: np.random.Generator, grid: int = 4, strength: float = 0.06) -> np.ndarray:
    """Warp with a grid x grid control lattice, each point jittered by N(0, strength * size).
    The image corners stay fixed so the warp doesn't pull in large border regions."""
    h, w = img.shape[:2]
    xs, ys = np.meshgrid(np.linspace(0, w - 1, grid), np.linspace(0, h - 1, grid))
    dst = np.stack([xs.ravel(), ys.ravel()], 1)
    src = dst + rng.normal(0, strength * min(h, w), dst.shape)
    corners = [0, grid - 1, grid * (grid - 1), grid * grid - 1]
    src[corners] = dst[corners]
    map_x, map_y = tps_grid(src, dst, h, w)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def random_crop_resize(img: np.ndarray, rng: np.random.Generator, min_scale: float = 0.8) -> np.ndarray:
    """Crop a random square covering min_scale..1 of the side, resized back to the input size."""
    h, w = img.shape[:2]
    side = int(round(min(h, w) * rng.uniform(min_scale, 1.0)))
    y0 = int(rng.integers(0, h - side + 1))
    x0 = int(rng.integers(0, w - side + 1))
    return cv2.resize(img[y0 : y0 + side, x0 : x0 + side], (w, h), interpolation=cv2.INTER_CUBIC)


def augment_reference(img: np.ndarray, rng: np.random.Generator, flip_prob: float = 0.5,
                      tps_strength: float = 0.06, min_crop: float = 0.8) -> np.ndarray:
    """(H,W,3) uint8 ground truth -> warped, maybe flipped, cropped reference of the same size."""
    out = random_tps(img, rng, strength=tps_strength)
    if rng.random() < flip_prob:
        out = out[:, ::-1]
    return np.ascontiguousarray(random_crop_resize(out, rng, min_crop))
