"""Shared Dataset for all three models.

Image ids are "<group>__<stem>", where <group> is a folder of the HF mirror
(ComputerVisionAnimeProject/AnimeFaceColorization):

    <root>/sketch/<group>/<stem>.{png,jpg}       line art, dark lines on white
    <root>/reference/<group>/<stem>.{png,jpg}    the colored image (our target Y)

Sketches go through one fixed clean-up (src/data/sketch.py) so all sources share one line style.

A group is one character (moeimouto, e.g. 000_hatsune_miku) or one AnimeDiffusion image with its
deformation-flow copies (AnimeDiffusion_1074: 1074, 1074_aug1, 1074_aug2).

Modes (how `reference` is built):
    train         random TPS warp + flip + crop of the target, new every time the sample is read
    self_ref      the same augmentation with a fixed per-image seed, so every model sees the same R
                  (fairness rule 5); export_eval_assets.py also saves it as PNG
    cross_ref     the target of another image of the same character (splits/cross_ref_pairs.csv)
    shuffled_ref  the target of another test image (splits/shuffled_ref_pairs.csv; prior-work FID protocol)
"""

import zlib
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.sketch import normalize_sketch
from src.data.warp import augment_reference

SIZE = 256
EXTS = (".png", ".jpg", ".jpeg", ".webp")
MODES = ("train", "self_ref", "cross_ref", "shuffled_ref")
SELF_REF_SEED = 2026  # with the image id, fixes each self_ref reference


def make_id(group: str, stem: str) -> str:
    return f"{group}__{stem}"


def split_id(image_id: str) -> tuple[str, str]:
    group, sep, stem = image_id.partition("__")
    if not sep:
        raise ValueError(f"image id {image_id!r} is not '<group>__<stem>'")
    return group, stem


def find_image(root: Path, kind: str, image_id: str) -> Path:
    group, stem = split_id(image_id)
    for ext in EXTS:
        path = root / kind / group / (stem + ext)
        if path.exists():
            return path
    raise FileNotFoundError(f"no {kind} image for {image_id!r} under {root / kind / group}")


def read_image(path: Path, gray: bool) -> np.ndarray:
    """uint8 (H,W) or (H,W,3) RGB at SIZE x SIZE."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"cannot read {path}")
    if not gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (SIZE, SIZE):
        shrink = img.shape[0] > SIZE
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_CUBIC)
    return img


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """uint8 (H,W) or (H,W,3) -> float (1|3,H,W) in [-1,1]."""
    t = torch.from_numpy(np.ascontiguousarray(img)).float() / 127.5 - 1
    return t[None] if t.ndim == 2 else t.permute(2, 0, 1).contiguous()


def fixed_rng(image_id: str, seed: int = SELF_REF_SEED) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(image_id.encode())])


class ColorizationDataset(Dataset):
    def __init__(self, root: str | Path, ids: list[str], mode: str, ref_of: dict[str, str] | None = None,
                 sketch_style: str = "binary"):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        if mode in ("cross_ref", "shuffled_ref"):
            if ref_of is None:
                raise ValueError(f"mode {mode!r} needs a sketch_id -> reference_id mapping")
            missing = [i for i in ids if i not in ref_of]
            if missing:
                raise KeyError(f"{len(missing)} ids have no {mode} reference, e.g. {missing[:3]}")
        self.root, self.ids, self.mode, self.ref_of = Path(root), list(ids), mode, ref_of
        self.sketch_style = sketch_style

    def __len__(self) -> int:
        return len(self.ids)

    def reference_image(self, image_id: str, target: np.ndarray) -> np.ndarray:
        if self.mode == "train":
            # Drawn from torch's RNG, which DataLoader seeds per worker and checkpoints restore.
            rng = np.random.default_rng(int(torch.randint(0, 2**62, (1,)).item()))
            return augment_reference(target, rng)
        if self.mode == "self_ref":
            return augment_reference(target, fixed_rng(image_id))
        return read_image(find_image(self.root, "reference", self.ref_of[image_id]), gray=False)

    def __getitem__(self, idx: int) -> dict:
        image_id = self.ids[idx]
        sketch = normalize_sketch(read_image(find_image(self.root, "sketch", image_id), gray=True), self.sketch_style)
        target = read_image(find_image(self.root, "reference", image_id), gray=False)
        return {"id": image_id, "sketch": to_tensor(sketch), "target": to_tensor(target),
                "reference": to_tensor(self.reference_image(image_id, target))}
