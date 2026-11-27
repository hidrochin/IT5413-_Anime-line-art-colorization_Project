from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.fake_data import make_pair

N_IDS = 4
SEEDS_USED = [0, 1, 2]


def write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


@pytest.fixture
def eval_tree(tmp_path: Path) -> Path:
    """tmp/data/eval/{gt,sketch,ref_*} and tmp/outputs/{good,noisy}/<mode>/<id>_s<seed>.png."""
    rng = np.random.default_rng(0)
    for i in range(N_IDS):
        image_id = f"img{i:03d}"
        color, sketch = make_pair(i)
        write_png(tmp_path / "data/eval/gt" / f"{image_id}.png", color)
        write_png(tmp_path / "data/eval/sketch" / f"{image_id}.png", sketch)
        write_png(tmp_path / "data/eval/ref_self_ref" / f"{image_id}.png", color[:, ::-1].copy())
        write_png(tmp_path / "data/eval/ref_cross_ref" / f"{image_id}.png", color)
        for mode in ("self_ref", "cross_ref"):
            for seed in SEEDS_USED:
                noise = rng.normal(0, 25, color.shape)
                noisy = np.clip(color + noise, 0, 255).astype(np.uint8)
                write_png(tmp_path / "outputs/good" / mode / f"{image_id}_s{seed}.png", color)
                write_png(tmp_path / "outputs/noisy" / mode / f"{image_id}_s{seed}.png", noisy)
                write_png(tmp_path / "outputs/other" / mode / f"{image_id}_s{seed}.png", make_pair(i + 50)[0])
    return tmp_path
