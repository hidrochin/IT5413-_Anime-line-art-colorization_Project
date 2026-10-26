"""Synthetic stand-ins for the shared Dataset and LPIPS, so tests need no downloads."""

import cv2
import numpy as np
import torch

SIZE = 256


def make_pair(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(color image (H,W,3) uint8, sketch (H,W) uint8): flat-colored ellipses with black outlines."""
    rng = np.random.default_rng(seed)
    color = np.full((SIZE, SIZE, 3), 255, np.uint8)
    sketch = np.full((SIZE, SIZE), 255, np.uint8)
    for _ in range(4):
        center = tuple(int(v) for v in rng.integers(60, 196, 2))
        axes = tuple(int(v) for v in rng.integers(20, 60, 2))
        fill = tuple(int(v) for v in rng.integers(0, 255, 3))
        cv2.ellipse(color, center, axes, 0, 0, 360, fill, -1)
        cv2.ellipse(color, center, axes, 0, 0, 360, (0, 0, 0), 2)
        cv2.ellipse(sketch, center, axes, 0, 0, 360, 0, 2)
    return color, sketch


def to_tensor(img: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img).float() / 127.5 - 1
    return t[None] if t.ndim == 2 else t.permute(2, 0, 1)


def build_dataset(split: str, mode: str, n: int = 4):
    samples = []
    for i in range(n):
        color, sketch = make_pair(i)
        ref = color[:, ::-1].copy() if mode == "self_ref" else make_pair(i + 100)[0]
        samples.append({"id": f"{split}{i:03d}", "sketch": to_tensor(sketch), "target": to_tensor(color),
                        "reference": to_tensor(ref)})
    return samples


class FakeLPIPS:
    """Mean absolute difference per image: zero for identical inputs, like LPIPS."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a.float() - b.float()).abs().flatten(1).mean(1)
