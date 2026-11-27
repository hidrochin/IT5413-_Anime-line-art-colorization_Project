"""PNG <-> tensor helpers shared by sampling, evaluation and figures."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor


def to_uint8(img: Tensor) -> np.ndarray:
    """(C,H,W) tensor in [-1,1] -> (H,W,C) or (H,W) uint8 array."""
    arr = ((img.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    arr = arr.permute(1, 2, 0).numpy()
    return arr[..., 0] if arr.shape[-1] == 1 else arr


def save_png(img: Tensor, path: str | Path) -> None:
    """Save a (C,H,W) tensor in [-1,1] as PNG, creating parent folders. Written to a temp file and renamed,
    so a session killed mid-write never leaves a truncated PNG that sample.py would then skip as done."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    Image.fromarray(to_uint8(img)).save(tmp, format="PNG")
    tmp.replace(path)


def load_png(path: str | Path, mode: str = "RGB") -> Tensor:
    """Load a PNG as a float tensor in [0,1]: (3,H,W) for RGB, (1,H,W) for L."""
    arr = np.asarray(Image.open(path).convert(mode), dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[..., None]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
