"""Tune Model C's sampling settings on the validation split (never on test; fairness rule 4).

    python -m scripts.tune_controlnet --config configs/controlnet.yaml

Coordinate search: sweep one parameter at a time (in `tune.order`), holding the others at the current
best, and keep the value with the best `tune.select_by` on `tune.num_images` val images (self_ref,
one seed). Every evaluated setting is appended to results/tune_controlnet.csv; rerunning skips
settings already in that file, so an interrupted Kaggle session resumes where it stopped. The winning
`model:` section is written to results/tune_controlnet_best.yaml.
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torchvision.utils import save_image
from tqdm import tqdm

from src.metrics.edges import edge_f1
from src.metrics.perceptual import LPIPS
from src.metrics.pixel import psnr, ssim
from src.models.controlnet import ControlNetColorizer, ControlNetConfig
from src.utils.image_io import to_uint8
from src.utils.registry import DEFAULT_DATASET, load_object

LOWER_IS_BETTER = {"lpips": True, "psnr": False, "ssim": False, "edge_f1": False}
REBUILD_KEYS = {"base_model"}


def load_val(dataset_spec: str, split: str, n: int) -> list[dict]:
    ds = load_object(dataset_spec)(split, "self_ref")
    return [ds[i] for i in range(min(n, len(ds)))]


def score(colorizer: ControlNetColorizer, samples: list[dict], seed: int, batch_size: int, lpips_fn,
          grid_path: Path | None = None) -> dict:
    preds, targets, sketches = [], [], []
    for k in tqdm(range(0, len(samples), batch_size), desc="val", leave=False):
        batch = samples[k : k + batch_size]
        sketch = torch.stack([s["sketch"] for s in batch])
        reference = torch.stack([s["reference"] for s in batch])
        preds.append(colorizer.colorize(sketch, reference, seed=seed).cpu())
        targets.append(torch.stack([s["target"] for s in batch]))
        sketches.append(sketch)
    pred, target, sketch = (torch.cat(x) for x in (preds, targets, sketches))
    p01, t01 = (pred + 1) / 2, (target + 1) / 2
    f1 = [edge_f1(to_uint8(p), to_uint8(s))[2] for p, s in zip(pred, sketch)]
    if grid_path is not None:
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        n = min(8, len(pred))
        save_image(torch.cat([sketch[:n].repeat(1, 3, 1, 1), target[:n], pred[:n]]) * 0.5 + 0.5, grid_path, nrow=n)
    return {
        "lpips": lpips_fn(p01, t01).mean().item(),
        "psnr": psnr(p01, t01).mean().item(),
        "ssim": ssim(p01, t01).mean().item(),
        "edge_f1": float(np.mean(f1)),
    }


def setting_key(setting: dict, keys: list[str]) -> str:
    return json.dumps({k: setting[k] for k in keys}, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/controlnet.yaml"))
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--num-images", type=int, default=None, help="override tune.num_images")
    parser.add_argument("--out", type=Path, default=Path("results/tune_controlnet.csv"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    tune = cfg["tune"]
    order, grid = tune["order"], tune["grid"]
    select_by = tune["select_by"]
    lower = LOWER_IS_BETTER[select_by]
    samples = load_val(args.dataset, tune["split"], args.num_images or tune["num_images"])

    done: dict[str, dict] = {}
    if args.out.exists():
        for row in pd.read_csv(args.out).to_dict("records"):
            done[row["key"]] = row
        print(f"resuming: {len(done)} settings already scored in {args.out}")

    current = dict(cfg["model"])
    lpips_fn = LPIPS(device=current["device"])
    colorizer, built_for = None, None

    for param in order:
        results = []
        for value in grid[param]:
            setting = {**current, param: value}
            key = setting_key(setting, order)
            if key not in done:
                rebuild_on = {k: setting[k] for k in REBUILD_KEYS}
                if colorizer is None or rebuild_on != built_for:
                    colorizer = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    colorizer = ControlNetColorizer(ControlNetConfig.from_dict(setting))
                    built_for = rebuild_on
                else:
                    colorizer.set_params(**{k: setting[k] for k in order if k not in REBUILD_KEYS})
                grid_path = args.out.parent / "figures" / "tune" / f"{len(done):03d}.png"
                metrics = score(colorizer, samples, tune["seed"], tune["batch_size"], lpips_fn, grid_path)
                row = {"key": key, "param": param, **{k: setting[k] for k in order}, **metrics,
                       "n_images": len(samples), "grid": str(grid_path)}
                done[key] = row
                args.out.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(list(done.values())).to_csv(args.out, index=False)
                print(f"{param}={value}: " + ", ".join(f"{m}={metrics[m]:.4f}" for m in metrics))
            results.append((done[key][select_by], value))
        best = (min if lower else max)(results, key=lambda r: r[0])[1]
        current[param] = best
        print(f"best {param} = {best}")

    best_path = args.out.parent / "tune_controlnet_best.yaml"
    best_path.write_text(yaml.safe_dump({"model": current}, sort_keys=False, allow_unicode=True))
    print(f"best setting written to {best_path}")


if __name__ == "__main__":
    main()
