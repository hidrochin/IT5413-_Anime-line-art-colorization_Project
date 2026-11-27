"""Visual check of the shared Dataset: rows of sketch / target / reference for N samples.

    python -m scripts.preview_dataset --split train --mode train --n 16
    python -m scripts.preview_dataset --split test --mode cross_ref --out results/figures/preview_cross_ref.png
"""

import argparse
import random
from pathlib import Path

import torch
from torchvision.utils import save_image

from src.utils.registry import DEFAULT_DATASET, load_object


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="train")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0, help="which samples to show")
    parser.add_argument("--out", type=Path, default=None, help="default: results/figures/preview_<split>_<mode>.png")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    args = parser.parse_args()

    ds = load_object(args.dataset)(args.split, args.mode)
    idx = sorted(random.Random(args.seed).sample(range(len(ds)), min(args.n, len(ds))))
    samples = [ds[i] for i in idx]
    rows = [torch.stack([s["sketch"].repeat(3, 1, 1) for s in samples]),
            torch.stack([s["target"] for s in samples]),
            torch.stack([s["reference"] for s in samples])]
    out = args.out or Path(f"results/figures/preview_{args.split}_{args.mode}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image(torch.cat(rows) * 0.5 + 0.5, out, nrow=len(samples), padding=2)
    print(f"rows: sketch / target / reference ({len(samples)} samples) -> {out}")


if __name__ == "__main__":
    main()
