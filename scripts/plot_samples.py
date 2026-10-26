"""Qualitative figures from saved PNGs.

    # Experiment 1: rows = test ids, columns = sketch | reference | [ground truth] | each model (seed 0)
    python -m scripts.plot_samples --models gan flow controlnet --mode self_ref --n 8
    # Diversity: rows = models, columns = the 5 seeds for one test id
    python -m scripts.plot_samples --models gan flow controlnet --mode self_ref --seeds-for <image_id>
"""

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src.metrics.files import index_by_id, index_outputs
from src.utils.seeding import SEEDS


def show(ax, path: Path | None, title: str = "") -> None:
    if path is not None:
        ax.imshow(Image.open(path).convert("RGB"))
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--mode", default="self_ref")
    parser.add_argument("--eval-dir", type=Path, default=Path("data/eval"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--ids", nargs="*", default=None, help="explicit test ids (default: random, seed 0)")
    parser.add_argument("--seeds-for", default=None, help="one test id: show every seed per model")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    outputs = {m: index_outputs(args.outputs_dir / m / args.mode) for m in args.models}
    sketch = index_by_id(args.eval_dir / "sketch")
    refs = index_by_id(args.eval_dir / f"ref_{args.mode}")
    gt = index_by_id(args.eval_dir / "gt") if args.mode == "self_ref" else {}

    if args.seeds_for:
        image_id = args.seeds_for
        fig, axes = plt.subplots(len(args.models), len(SEEDS) + 2, figsize=(2 * (len(SEEDS) + 2), 2 * len(args.models)), squeeze=False)
        for r, model in enumerate(args.models):
            show(axes[r][0], sketch[image_id], "sketch" if r == 0 else "")
            show(axes[r][1], refs[image_id], "reference" if r == 0 else "")
            for c, seed in enumerate(SEEDS):
                show(axes[r][c + 2], outputs[model].get(image_id, {}).get(seed), f"seed {seed}" if r == 0 else "")
            axes[r][0].set_ylabel(model)
        out = args.out or Path(f"results/figures/diversity_{args.mode}_{image_id}.png")
    else:
        common = sorted(set(sketch).intersection(*[set(o) for o in outputs.values()]))
        ids = args.ids or random.Random(0).sample(common, min(args.n, len(common)))
        cols = ["sketch", "reference"] + (["ground truth"] if gt else []) + args.models
        fig, axes = plt.subplots(len(ids), len(cols), figsize=(2 * len(cols), 2 * len(ids)), squeeze=False)
        for r, image_id in enumerate(ids):
            paths = [sketch[image_id], refs.get(image_id)] + ([gt[image_id]] if gt else [])
            paths += [outputs[m].get(image_id, {}).get(SEEDS[0]) for m in args.models]
            for c, (col, path) in enumerate(zip(cols, paths)):
                show(axes[r][c], path, col if r == 0 else "")
        out = args.out or Path(f"results/figures/samples_{args.mode}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
