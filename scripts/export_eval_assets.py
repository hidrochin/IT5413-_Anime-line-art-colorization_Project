"""Dump a split's ground truth, sketches and fixed references as PNGs, so evaluate.py only reads PNGs.

    python -m scripts.export_eval_assets --split test --out data/eval

Writes <out>/gt/<id>.png, <out>/sketch/<id>.png and <out>/ref_<mode>/<id>.png for each mode. The
reference in ref_<mode>/<id>.png is exactly the one the models receive for sketch <id> in that mode.
"""

import argparse
from pathlib import Path

from tqdm import tqdm

from src.utils.image_io import save_png
from src.utils.registry import DEFAULT_DATASET, load_object


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, default=Path("data/eval"))
    parser.add_argument("--modes", nargs="+", default=["self_ref", "cross_ref"])
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="factory 'module:attr' called as f(split, mode)")
    args = parser.parse_args()

    build_dataset = load_object(args.dataset)
    for i, mode in enumerate(args.modes):
        ds = build_dataset(args.split, mode)
        for idx in tqdm(range(len(ds)), desc=f"{args.split}/{mode}"):
            sample = ds[idx]
            image_id = sample["id"]
            if i == 0:
                save_png(sample["target"], args.out / "gt" / f"{image_id}.png")
                save_png(sample["sketch"], args.out / "sketch" / f"{image_id}.png")
            save_png(sample["reference"], args.out / f"ref_{mode}" / f"{image_id}.png")
    print(f"wrote eval assets to {args.out}")


if __name__ == "__main__":
    main()
