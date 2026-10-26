"""Generate test outputs for any model through the shared Colorizer interface (owner: Thành).

    python -m scripts.sample --config configs/gan.yaml --mode self_ref cross_ref shuffled_ref
    python -m scripts.sample --config results/tune_controlnet_best.yaml --model controlnet --mode self_ref
    python -m scripts.sample --config configs/flow.yaml --steps 8 --mode self_ref          # run "flow_steps8"
    python -m scripts.sample --config configs/controlnet.yaml --model controlnet_ip0.8 --set ip_adapter_scale=0.8

Writes outputs/<model>/<mode>/<image_id>_s<seed>.png for every id of the mode's test split
(src.data.eval_split: cross_ref uses test_cross, the others test) x SEEDS, and appends one row
per mode to results/speed.csv (model,steps,nfe,sec_per_image,gpu). Images that already exist are
skipped, so an interrupted Kaggle session can simply be rerun.

The Colorizer class is the config's `colorizer: module:Class` if given, else chosen by the run name's
prefix (see COLORIZERS). It is built with `Class.from_config(config["model"])`.
"""

import argparse
import time
import warnings
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.colorizer import Colorizer
from src.data import eval_split
from src.utils.image_io import save_png
from src.utils.registry import DEFAULT_DATASET, load_object
from src.utils.seeding import SEEDS

COLORIZERS = {
    "gan": "src.models.gan:GanColorizer",
    "flow": "src.models.flow:FlowColorizer",
    "controlnet": "src.models.controlnet:ControlNetColorizer",
}
MODES = ["self_ref", "cross_ref", "shuffled_ref"]
SPEED_COLUMNS = ["model", "steps", "nfe", "sec_per_image", "gpu"]


def resolve_colorizer(run_name: str, spec: str | None) -> type[Colorizer]:
    if spec is None:
        prefixes = [p for p in COLORIZERS if run_name == p or run_name.startswith(p + "_")]
        if not prefixes:
            raise ValueError(f"cannot pick a Colorizer for run {run_name!r}; set `colorizer: module:Class` "
                             f"in the config or pass --colorizer (known prefixes: {sorted(COLORIZERS)})")
        spec = COLORIZERS[max(prefixes, key=len)]
    cls = load_object(spec)
    if not (isinstance(cls, type) and issubclass(cls, Colorizer)):
        raise TypeError(f"{spec} is not a Colorizer subclass")
    return cls


def parse_overrides(pairs: list[str]) -> dict:
    """['ip_adapter_scale=0.8', 'lora_path=null'] -> {'ip_adapter_scale': 0.8, 'lora_path': None}."""
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"--set expects key=value, got {pair!r}")
        out[key] = yaml.safe_load(value)
    return out


def run_name_for(cfg: dict, model: str | None, steps: int | None) -> str:
    if model:
        return model
    name = cfg.get("name")
    if name is None:
        raise ValueError("config has no `name:`; pass --model")
    return f"{name}_steps{steps}" if steps is not None else name


def gpu_name() -> str:
    return torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def sample_mode(colorizer: Colorizer, dataset, out_dir: Path, seeds: list[int], steps: int | None,
                batch_size: int, num_workers: int, device: str, overwrite: bool) -> tuple[int, float]:
    """Write every missing <id>_s<seed>.png. Returns (images timed, seconds) for the speed estimate;
    the first colorize call is warm-up (lazy init, cuDNN autotune) and only counts if it is the only one."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    timings: list[tuple[int, float]] = []
    for batch in tqdm(loader, desc=out_dir.parent.name + "/" + out_dir.name):
        ids = list(batch["id"])
        sketch = batch["sketch"].to(device)
        reference = batch["reference"].to(device)
        for seed in seeds:
            paths = [out_dir / f"{i}_s{seed}.png" for i in ids]
            if not overwrite and all(p.exists() for p in paths):
                continue
            sync()
            t0 = time.perf_counter()
            out = colorizer.colorize(sketch, reference, seed=seed, steps=steps)
            sync()
            dt = time.perf_counter() - t0
            expected = (len(ids), 3, *sketch.shape[-2:])
            if tuple(out.shape) != expected:
                raise ValueError(f"{type(colorizer).__name__}.colorize returned {tuple(out.shape)}, expected {expected}")
            timings.append((len(ids), dt))
            for img, path in zip(out.clamp(-1, 1).cpu(), paths):
                save_png(img, path)
    timed = timings[1:] or timings
    return sum(n for n, _ in timed), sum(t for _, t in timed)


def append_speed(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row], columns=SPEED_COLUMNS)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="YAML with a `model:` section")
    parser.add_argument("--model", default=None, help="run name / output folder (default: config `name`, "
                        "plus _steps<N> when --steps is given)")
    parser.add_argument("--colorizer", default=None, help="'module:Class' (overrides the config's `colorizer:`)")
    parser.add_argument("--mode", nargs="+", default=["self_ref"], choices=MODES)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override `model:` entries")
    parser.add_argument("--steps", type=int, default=None, help="sampling steps (None = model default)")
    parser.add_argument("--split", default=None, help="default: each mode's test split (cross_ref -> test_cross)")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="only the first N ids (smoke tests)")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="factory 'module:attr' called as f(split, mode)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not set(args.seeds) <= set(SEEDS):
        warnings.warn(f"--seeds {args.seeds} are not all in SEEDS={SEEDS}; outputs won't match other models")
    if args.split is None and args.limit is None and args.seeds != SEEDS:
        warnings.warn("full test run with fewer seeds than SEEDS: diversity and FID mean/std will differ")

    cfg = yaml.safe_load(args.config.read_text())
    model_cfg = {**cfg.get("model", {}), **parse_overrides(args.set)}
    run = run_name_for(cfg, args.model, args.steps)
    cls = resolve_colorizer(run, args.colorizer or cfg.get("colorizer"))
    colorizer = cls.from_config(model_cfg)
    build_dataset = load_object(args.dataset)
    steps_used = args.steps if args.steps is not None else model_cfg.get("steps", 1)

    for mode in args.mode:
        dataset = build_dataset(args.split or eval_split(mode), mode)
        if args.limit is not None:
            dataset = torch.utils.data.Subset(dataset, range(min(args.limit, len(dataset))))
        out_dir = args.outputs_dir / run / mode
        n, sec = sample_mode(colorizer, dataset, out_dir, args.seeds, args.steps, args.batch_size,
                             args.num_workers, args.device, args.overwrite)
        if n == 0:
            print(f"{out_dir}: all outputs already exist")
            continue
        row = {"model": run, "steps": steps_used, "nfe": colorizer.nfe(args.steps),
               "sec_per_image": round(sec / n, 5), "gpu": gpu_name()}
        append_speed(args.results_dir / "speed.csv", row)
        print(f"{out_dir}: {row['sec_per_image']:.3f} s/image on {row['gpu']}")


if __name__ == "__main__":
    main()
