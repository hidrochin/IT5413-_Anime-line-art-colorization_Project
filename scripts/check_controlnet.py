"""Sanity checks for Model C on a few val images (never test): does the output follow *its own* reference?

    python -m scripts.check_controlnet --config configs/controlnet.yaml [--num-images 8] [--set key=value ...] [--name plus]

Prints, and appends to results/check_controlnet.csv:
    batch_diff      mean |output in a batch - same sample alone| (in [-1,1]); ~0 means samples don't leak into
                    each other (one reference per sample, fairness of the seed contract)
    match_ip0       how often an output's Lab color histogram is closest to its own reference among all n
                    references, with the IP-Adapter off (scale 0): the baseline from sketch + prompt alone
    match           the same with the IP-Adapter on (configured scale)
    match_swapped   references rolled by one image: how often the output is closest to the reference it was
                    *given* (not its own ground truth). Chance level for the three match rates is 1/n.
Grid (results/figures/check_controlnet_<name>.png), one row per image:
    sketch | reference | IP off | IP on | swapped reference | output with it | ground truth
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torchvision.utils import save_image

from scripts.sample import parse_overrides
from src.metrics.reference import color_distance
from src.models.controlnet import ControlNetColorizer, ControlNetConfig
from src.utils.image_io import to_uint8
from src.utils.registry import DEFAULT_DATASET, load_object


def colorize_batched(colorizer, sketch, reference, batch: int, seed: int) -> torch.Tensor:
    return torch.cat([colorizer.colorize(sketch[k : k + batch], reference[k : k + batch], seed=seed).cpu()
                      for k in range(0, len(sketch), batch)])


def match_rate(outputs: torch.Tensor, references: torch.Tensor, given: list[int]) -> float:
    """Fraction of outputs whose color histogram is closest to references[given[i]] among all references."""
    refs = [to_uint8(r) for r in references]
    hits = 0
    for i, out in enumerate(outputs):
        d = [color_distance(to_uint8(out), r) for r in refs]
        hits += int(np.argmin(d) == given[i])
    return hits / len(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/controlnet.yaml"))
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override `model:` entries")
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="default", help="label for the CSV row and the grid file")
    parser.add_argument("--split", default="val")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.split.startswith("test"):
        raise ValueError("checks run on val only (fairness rule 4)")

    cfg = yaml.safe_load(args.config.read_text())
    model_cfg = {**cfg["model"], **parse_overrides(args.set)}
    ds = load_object(args.dataset)(args.split, "self_ref")
    items = [ds[i] for i in range(min(args.num_images, len(ds)))]
    n = len(items)
    sketch = torch.stack([x["sketch"] for x in items])
    reference = torch.stack([x["reference"] for x in items])
    target = torch.stack([x["target"] for x in items])
    own, rolled = list(range(n)), [(i + 1) % n for i in range(n)]

    colorizer = ControlNetColorizer(ControlNetConfig.from_dict(model_cfg))
    scale = colorizer.cfg.ip_adapter_scale
    out = colorize_batched(colorizer, sketch, reference, args.batch_size, args.seed)
    alone = colorize_batched(colorizer, sketch, reference, 1, args.seed)
    swapped = colorize_batched(colorizer, sketch, reference[rolled], args.batch_size, args.seed)
    colorizer.set_params(ip_adapter_scale=0.0)
    ip0 = colorize_batched(colorizer, sketch, reference, args.batch_size, args.seed)
    colorizer.set_params(ip_adapter_scale=scale)

    row = {"name": args.name, "n": n, **{k: model_cfg[k] for k in ("ip_adapter_weight", "ip_adapter_scale",
                                                                    "controlnet_scale", "guidance_scale", "steps")},
           "batch_diff": (out - alone).abs().mean().item(),
           "match_ip0": match_rate(ip0, reference, own),
           "match": match_rate(out, reference, own),
           "match_swapped": match_rate(swapped, reference, rolled),
           "chance": 1 / n}
    for k, v in row.items():
        print(f"{k:>18}: {v:.4f}" if isinstance(v, float) else f"{k:>18}: {v}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.results_dir / "check_controlnet.csv"
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
    cols = [sketch.repeat(1, 3, 1, 1), reference, ip0, out, reference[rolled], swapped, target]
    grid = torch.stack([c[i] for i in range(n) for c in cols])
    grid_path = args.results_dir / "figures" / f"check_controlnet_{args.name}.png"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid * 0.5 + 0.5, grid_path, nrow=len(cols))
    print(f"grid -> {grid_path}, row appended to {csv_path}")


if __name__ == "__main__":
    main()
