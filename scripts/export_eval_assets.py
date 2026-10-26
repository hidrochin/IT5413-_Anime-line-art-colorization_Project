"""Dump the test sets' ground truth, sketches and fixed references as PNGs, so evaluate.py only reads PNGs.

    python -m scripts.export_eval_assets --out data/eval

Writes <out>/gt/<id>.png, <out>/sketch/<id>.png and <out>/ref_<mode>/<id>.png for each mode. Each mode
uses its own test split (src.data.eval_split: self_ref/shuffled_ref -> test, cross_ref -> test_cross),
so gt/ and sketch/ hold both test sets, and ref_<mode>/ lists exactly the ids scored in that mode. The
reference in ref_<mode>/<id>.png is exactly the one the models receive for sketch <id> in that mode.
"""

import argparse
from pathlib import Path

from tqdm import tqdm

from src.data import MODE_SPLIT, eval_split
from src.utils.image_io import save_png
from src.utils.registry import DEFAULT_DATASET, load_object


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default=None, help="force one split for every mode (default: per-mode test split)")
    parser.add_argument("--out", type=Path, default=Path("data/eval"))
    parser.add_argument("--modes", nargs="+", default=list(MODE_SPLIT))
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="factory 'module:attr' called as f(split, mode)")
    args = parser.parse_args()

    build_dataset = load_object(args.dataset)
    for mode in args.modes:
        split = args.split or eval_split(mode)
        ds = build_dataset(split, mode)
        for idx in tqdm(range(len(ds)), desc=f"{split}/{mode}"):
            sample = ds[idx]
            image_id = sample["id"]
            for kind, key in (("gt", "target"), ("sketch", "sketch")):
                path = args.out / kind / f"{image_id}.png"
                if not path.exists():
                    save_png(sample[key], path)
            save_png(sample["reference"], args.out / f"ref_{mode}" / f"{image_id}.png")
    print(f"wrote eval assets to {args.out}")


if __name__ == "__main__":
    main()
