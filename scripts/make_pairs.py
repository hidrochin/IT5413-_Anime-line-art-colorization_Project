"""Write splits/cross_ref_pairs.csv (for test_cross) and splits/shuffled_ref_pairs.csv (for test). Owner: Cường.

    python -m scripts.make_pairs --root data/AnimeFaceColorization

cross_ref: for each test_cross sketch, the colored image of *another drawing* of the same character,
preferring drawings that are not test_cross sketches themselves ("other_image"). Only if a group has
no other drawing does it fall back to a deformation-flow copy of the same drawing ("deformation",
close to self-reference); the `kind` column records which, and the script warns about any fallback.

shuffled_ref: each test sketch gets the colored image of a different test image (a seeded derangement),
the protocol AnimeDiffusion used for its FID.
"""

import argparse
import csv
import random
from pathlib import Path

from scripts.make_splits import is_aug, scan_pairs
from src.data import read_ids, split_id, eval_split
from src.data.dataset import make_id


def cross_ref_pairs(sketch_ids: list[str], groups: dict[str, list[str]], seed: int) -> list[dict]:
    rng = random.Random(seed)
    taken = set(sketch_ids)
    rows = []
    for image_id in sketch_ids:
        group, stem = split_id(image_id)
        others = [s for s in groups.get(group, []) if s != stem]
        drawings = [s for s in others if not is_aug(s)]
        spare = [s for s in drawings if make_id(group, s) not in taken]
        if spare or drawings:
            rows.append({"sketch_id": image_id, "reference_id": make_id(group, rng.choice(spare or drawings)),
                         "kind": "other_image"})
        elif others:
            rows.append({"sketch_id": image_id, "reference_id": make_id(group, rng.choice(others)), "kind": "deformation"})
        else:
            raise SystemExit(f"{image_id}: no other image in group {group!r} to use as a cross reference")
    return rows


def shuffled_ref_pairs(test_ids: list[str], seed: int) -> list[dict]:
    order = sorted(test_ids)
    random.Random(seed).shuffle(order)
    # Pair each id with the next one in a shuffled cycle: a derangement, so nobody gets their own image.
    return sorted(({"sketch_id": a, "reference_id": b} for a, b in zip(order, order[1:] + order[:1])),
                  key=lambda r: r["sketch_id"])


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/AnimeFaceColorization"))
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="overwrite existing pair files")
    args = parser.parse_args()

    targets = {"cross": args.splits_dir / "cross_ref_pairs.csv", "shuffled": args.splits_dir / "shuffled_ref_pairs.csv"}
    if any(p.exists() for p in targets.values()) and not args.force:
        raise SystemExit(f"pair files already exist in {args.splits_dir}; use --force only if the team agreed")
    cross = cross_ref_pairs(read_ids(eval_split("cross_ref"), args.splits_dir), scan_pairs(args.root), args.seed)
    test_ids = read_ids(eval_split("shuffled_ref"), args.splits_dir)
    write_csv(targets["cross"], cross)
    write_csv(targets["shuffled"], shuffled_ref_pairs(test_ids, args.seed))
    n_def = sum(r["kind"] == "deformation" for r in cross)
    print(f"cross_ref: {len(cross)} pairs ({len(cross) - n_def} other_image, {n_def} deformation)")
    if n_def:
        print(f"warning: {n_def}/{len(cross)} cross_ref references are deformation copies of the same drawing")
    print(f"shuffled_ref: {len(test_ids)} pairs")


if __name__ == "__main__":
    main()
