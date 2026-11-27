"""Shared Dataset, sketch clean-up, reference warp, split and pair generation on a tiny fake mirror."""

import sys

import cv2
import numpy as np
import pytest
import torch

import scripts.make_pairs as make_pairs
import scripts.make_splits as make_splits
from src.data import build_dataset, read_ids, read_pairs, eval_split
from src.data.sketch import binarize, is_binary
from src.data.warp import augment_reference, tps_grid
from tests.fake_data import make_pair, write_mirror


@pytest.fixture
def mirror(tmp_path, monkeypatch):
    """Fake mirror + splits (2 test, 2 val AnimeDiffusion groups; 1 held-out character x 2 test_cross) + pairs."""
    root, splits = tmp_path / "mirror", tmp_path / "splits"
    write_mirror(root)
    monkeypatch.setattr(sys, "argv", ["make_splits", "--root", str(root), "--out", str(splits), "--n-test", "2", "--n-val", "2",
                                      "--cross-chars", "1", "--cross-per-char", "2"])
    make_splits.main()
    monkeypatch.setattr(sys, "argv", ["make_pairs", "--root", str(root), "--splits-dir", str(splits)])
    make_pairs.main()
    return root, splits


def test_tps_identity_and_augment():
    pts = np.array([[0, 0], [255, 0], [0, 255], [255, 255], [128, 128]], float)
    mx, my = tps_grid(pts, pts, 256, 256)
    ys, xs = np.mgrid[0:256, 0:256]
    assert np.abs(mx - xs).max() < 1e-3 and np.abs(my - ys).max() < 1e-3

    img = make_pair(0)[0]
    a = augment_reference(img, np.random.default_rng(1))
    b = augment_reference(img, np.random.default_rng(1))
    assert a.shape == img.shape and a.dtype == np.uint8
    assert np.array_equal(a, b) and not np.array_equal(a, img)


def test_splits_keep_groups_together(mirror):
    _, splits = mirror
    ids = {s: read_ids(s, splits) for s in ("train", "val", "test", "test_cross")}
    assert len(ids["test"]) == 2 and len(ids["val"]) == 2 and len(ids["test_cross"]) == 2
    groups = {s: {i.split("__")[0] for i in v} for s, v in ids.items()}
    for held in ("val", "test", "test_cross"):
        assert not groups["train"] & groups[held]
    assert len(groups["test_cross"]) == 1 and not next(iter(groups["test_cross"])).startswith("AnimeDiffusion_")
    assert not any("_aug" in i for i in ids["test"] + ids["val"])
    assert all(g.startswith("AnimeDiffusion_") for g in groups["test"] | groups["val"])
    # 1 remaining character x 3 drawings + 2 remaining AnimeDiffusion groups x 3 images
    assert len(ids["train"]) == 3 + 2 * 3


def test_splits_refuse_overwrite(mirror, monkeypatch):
    root, splits = mirror
    monkeypatch.setattr(sys, "argv", ["make_splits", "--root", str(root), "--out", str(splits), "--n-test", "2", "--n-val", "2"])
    with pytest.raises(SystemExit, match="already exist"):
        make_splits.main()


def test_pairs(mirror):
    _, splits = mirror
    test, test_cross = read_ids("test", splits), read_ids("test_cross", splits)
    cross, shuffled = read_pairs("cross_ref", splits), read_pairs("shuffled_ref", splits)
    assert set(cross) == set(test_cross)
    # the held-out character has 3 drawings, 2 are test sketches -> both take the spare one
    assert len(set(cross.values()) - set(test_cross)) == 1
    for sketch_id, ref_id in cross.items():
        assert ref_id != sketch_id and ref_id.split("__")[0] == sketch_id.split("__")[0]
    assert set(shuffled) == set(test) == set(shuffled.values())
    assert all(k != v for k, v in shuffled.items())


def test_cross_ref_prefers_other_drawing():
    groups = {"001_x": ["face_0", "face_1", "face_2", "face_0_aug1"], "AnimeDiffusion_3": ["3", "3_aug1"]}
    rows = make_pairs.cross_ref_pairs(["001_x__face_0", "001_x__face_1", "AnimeDiffusion_3__3"], groups, seed=0)
    assert [r["reference_id"] for r in rows[:2]] == ["001_x__face_2"] * 2  # the drawing that isn't a test sketch
    assert rows[0]["kind"] == "other_image"
    assert rows[2]["reference_id"] == "AnimeDiffusion_3__3_aug1" and rows[2]["kind"] == "deformation"


@pytest.mark.parametrize("mode", ["train", "self_ref", "cross_ref", "shuffled_ref"])
def test_dataset_modes(mirror, mode):
    root, splits = mirror
    split = "train" if mode == "train" else eval_split(mode)
    ds = build_dataset(split, mode, root=root, splits_dir=splits)
    s = ds[0]
    assert s["id"] == read_ids(split, splits)[0]
    assert s["sketch"].shape == (1, 256, 256) and s["target"].shape == s["reference"].shape == (3, 256, 256)
    for key in ("sketch", "target", "reference"):
        assert s[key].min() >= -1 and s[key].max() <= 1
    assert not torch.equal(s["reference"], s["target"])


def test_self_ref_is_fixed_and_train_is_random(mirror):
    root, splits = mirror
    fixed = build_dataset("test", "self_ref", root=root, splits_dir=splits)
    assert torch.equal(fixed[0]["reference"], fixed[0]["reference"])
    train = build_dataset("train", "train", root=root, splits_dir=splits)
    assert not torch.equal(train[0]["reference"], train[0]["reference"])


def test_dataset_batches_with_default_collate(mirror):
    root, splits = mirror
    batch = next(iter(torch.utils.data.DataLoader(build_dataset("train", "train", root=root, splits_dir=splits), batch_size=4)))
    assert batch["sketch"].shape == (4, 1, 256, 256) and len(batch["id"]) == 4


def test_binarize_keeps_binary_sketches_and_cleans_gray_ones():
    binary = make_pair(0)[1]
    assert is_binary(binary) and np.array_equal(binarize(binary), binary)
    gray = (255 - (255 - binary) // 4).astype(np.uint8)  # faint lines at ~192: no pixel below the dark threshold
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    out = binarize(gray)
    assert not is_binary(gray) and set(np.unique(out)) <= {0, 255}
    lines = binary < 128
    assert (out[lines] == 0).mean() > 0.8 and (out[~lines] == 255).mean() > 0.9


def test_sketch_style_switch(mirror):
    root, splits = mirror
    raw = build_dataset("test", "self_ref", root=root, splits_dir=splits, sketch_style="raw")[0]["sketch"]
    binary = build_dataset("test", "self_ref", root=root, splits_dir=splits)[0]["sketch"]
    assert set(binary.unique().tolist()) <= {-1.0, 1.0} and raw.shape == binary.shape
    with pytest.raises(ValueError, match="sketch style"):
        build_dataset("test", "self_ref", root=root, splits_dir=splits, sketch_style="fancy")[0]
