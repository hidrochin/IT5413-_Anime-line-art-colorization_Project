"""Write splits/{train,val,test,test_cross}.txt once, with a fixed seed (fairness rule 3).

    python -m scripts.make_splits --root data/AnimeFaceColorization
    python -m scripts.make_splits --n-test 10 --n-val 10 --cross-chars 2 --cross-per-char 5 --out data/sample_splits

Splits are made by *group* (folder), so an image and its deformation-flow copies (<n>_aug1, <n>_aug2)
never land in different splits.

- test / val: groups drawn from `--test-pool` (AnimeDiffusion images by default: the benchmark the
  number-to-beat was measured on). Only their original image is used; their _aug copies are dropped.
- test_cross: `--cross-chars` whole characters from `--cross-pool` (moeimouto) are held out of training
  entirely; `--cross-per-char` drawings of each become cross_ref test sketches, and make_pairs.py takes
  the reference from another drawing of the same character. A character needs more drawings than that.
- train: every other pair.

When the official AnimeDiffusion split arrives, pass it with --test-ids so the 579 test images match
the paper exactly. Existing split files are never overwritten without --force.
"""

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path

from src.data.dataset import EXTS, make_id

AUG = re.compile(r"_aug\d+$")
SPLITS = ("train", "val", "test", "test_cross")


def scan_pairs(root: Path) -> dict[str, list[str]]:
    """{group: [stem, ...]} for every image that has both a sketch and a colored version."""
    def stems(kind: str) -> dict[str, set[str]]:
        out = defaultdict(set)
        for group_dir in (root / kind).iterdir():
            if group_dir.is_dir():
                out[group_dir.name] = {p.stem for p in group_dir.iterdir() if p.suffix.lower() in EXTS}
        return out

    if not (root / "sketch").is_dir() or not (root / "reference").is_dir():
        raise FileNotFoundError(f"{root} must contain sketch/ and reference/ folders")
    sketch, color = stems("sketch"), stems("reference")
    pairs = {g: sorted(sketch[g] & color[g]) for g in sorted(set(sketch) & set(color))}
    return {g: s for g, s in pairs.items() if s}


def is_aug(stem: str) -> bool:
    return AUG.search(stem) is not None


def make_splits(groups: dict[str, list[str]], n_test: int, n_val: int, test_pool: str, seed: int,
                cross_chars: int = 0, cross_per_char: int = 0, cross_pool: str = r"^\d{3}_",
                train_aug: bool = True, test_ids: list[str] | None = None) -> dict[str, list[str]]:
    rng = random.Random(seed)
    pool = sorted(g for g in groups if re.search(test_pool, g))
    rng.shuffle(pool)
    if test_ids is not None:
        test_groups = {i.partition("__")[0] for i in test_ids}
        pool = [g for g in pool if g not in test_groups]
        held = {"test": sorted(test_groups), "val": pool[:n_val]}
    else:
        if len(pool) < n_test + n_val:
            raise SystemExit(f"only {len(pool)} groups match --test-pool {test_pool!r}; need {n_test + n_val}")
        held = {"test": pool[:n_test], "val": pool[n_test : n_test + n_val]}

    chars = sorted(g for g in groups if re.search(cross_pool, g) and g not in held["test"] + held["val"]
                   and sum(not is_aug(s) for s in groups[g]) > cross_per_char)
    if len(chars) < cross_chars:
        raise SystemExit(f"only {len(chars)} groups match --cross-pool {cross_pool!r} with more than "
                         f"{cross_per_char} drawings; need {cross_chars}")
    rng.shuffle(chars)
    held["test_cross"] = chars[:cross_chars]

    def originals(gs):
        return [make_id(g, s) for g in gs for s in groups[g] if not is_aug(s)]

    out = {"test": sorted(test_ids) if test_ids is not None else originals(held["test"]),
           "val": originals(held["val"]),
           "test_cross": [make_id(g, s) for g in held["test_cross"]
                          for s in rng.sample(sorted(s for s in groups[g] if not is_aug(s)), cross_per_char)]}
    held_out = {g for gs in held.values() for g in gs}
    out["train"] = [make_id(g, s) for g in groups if g not in held_out
                    for s in groups[g] if train_aug or not is_aug(s)]
    return {k: sorted(v) for k, v in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/AnimeFaceColorization"))
    parser.add_argument("--out", type=Path, default=Path("splits"))
    parser.add_argument("--n-test", type=int, default=579)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--test-pool", default=r"^AnimeDiffusion_", help="regex on group names eligible for val/test")
    parser.add_argument("--cross-chars", type=int, default=40, help="characters held out for test_cross")
    parser.add_argument("--cross-per-char", type=int, default=15, help="test_cross sketches per held-out character")
    parser.add_argument("--cross-pool", default=r"^\d{3}_", help="regex on group names eligible for test_cross")
    parser.add_argument("--test-ids", type=Path, default=None, help="file with the official test ids, one per line")
    parser.add_argument("--no-train-aug", action="store_true", help="leave the _aug deformation copies out of train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="overwrite existing split files")
    args = parser.parse_args()

    targets = [args.out / f"{s}.txt" for s in SPLITS]
    if any(p.exists() for p in targets) and not args.force:
        raise SystemExit(f"split files already exist in {args.out}; they are committed and shared. Use --force only "
                         "if the whole team agreed to regenerate them")
    test_ids = args.test_ids.read_text().split() if args.test_ids else None
    splits = make_splits(scan_pairs(args.root), args.n_test, args.n_val, args.test_pool, args.seed,
                         args.cross_chars, args.cross_per_char, args.cross_pool, not args.no_train_aug, test_ids)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        ids = splits[name]
        (args.out / f"{name}.txt").write_text("\n".join(ids) + "\n")
        print(f"{name}: {len(ids)} images ({len({i.partition('__')[0] for i in ids})} groups)")


if __name__ == "__main__":
    main()
