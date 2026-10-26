"""Shared data entry point: `build_dataset(split, mode)` (the interface every script calls).

The data root is $ANIME_DATA_ROOT, else data/AnimeFaceColorization (on Kaggle, point the env var at
the attached dataset, e.g. /kaggle/input/anime-face-colorization). Splits and pair files come from
$ANIME_SPLITS_DIR, else the committed splits/ (scripts/make_splits.py, scripts/make_pairs.py); point
it at a scratch folder (e.g. data/sample_splits) for smoke tests on a local sample. The sketch clean-up
is $ANIME_SKETCH_STYLE (binary by default, raw for the ablation; see src/data/sketch.py).

Two test sets: `test` (AnimeDiffusion images, for self_ref and shuffled_ref, comparable with prior
work) and `test_cross` (held-out moeimouto characters, whose other drawings are real cross references).
`eval_split(mode)` says which one a mode uses; sample.py and export_eval_assets.py follow it.
"""

import csv
import os
from pathlib import Path

from src.data.dataset import ColorizationDataset, make_id, split_id

DEFAULT_ROOT = Path("data/AnimeFaceColorization")
SPLITS_DIR = Path(os.environ.get("ANIME_SPLITS_DIR", "splits"))
PAIR_FILES = {"cross_ref": "cross_ref_pairs.csv", "shuffled_ref": "shuffled_ref_pairs.csv"}
MODE_SPLIT = {"self_ref": "test", "shuffled_ref": "test", "cross_ref": "test_cross"}

__all__ = ["ColorizationDataset", "MODE_SPLIT", "build_dataset", "data_root", "make_id", "read_ids", "read_pairs",
           "split_id", "eval_split"]


def data_root() -> Path:
    return Path(os.environ.get("ANIME_DATA_ROOT", DEFAULT_ROOT))


def eval_split(mode: str) -> str:
    """The test split a mode is evaluated on."""
    if mode not in MODE_SPLIT:
        raise ValueError(f"no test split for mode {mode!r}; expected one of {sorted(MODE_SPLIT)}")
    return MODE_SPLIT[mode]


def read_ids(split: str, splits_dir: Path = SPLITS_DIR) -> list[str]:
    path = splits_dir / f"{split}.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m scripts.make_splits` first")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def read_pairs(mode: str, splits_dir: Path = SPLITS_DIR) -> dict[str, str]:
    path = splits_dir / PAIR_FILES[mode]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m scripts.make_pairs` first")
    with path.open(newline="") as f:
        return {row["sketch_id"]: row["reference_id"] for row in csv.DictReader(f)}


def build_dataset(split: str, mode: str, root: str | Path | None = None,
                  splits_dir: str | Path = SPLITS_DIR, sketch_style: str | None = None) -> ColorizationDataset:
    splits_dir = Path(splits_dir)
    ref_of = read_pairs(mode, splits_dir) if mode in PAIR_FILES else None
    style = sketch_style or os.environ.get("ANIME_SKETCH_STYLE", "binary")
    return ColorizationDataset(root or data_root(), read_ids(split, splits_dir), mode, ref_of, style)
