"""Score one run from its PNGs. Reads only PNG folders and never imports model code (fairness rule 6).

    python -m scripts.evaluate --model gan --mode self_ref
    python -m scripts.evaluate --model controlnet --mode cross_ref --metrics fid,kid,color_dist

Inputs
    outputs/<model>/<mode>/<id>_s<seed>.png           written by sample.py
    <eval-dir>/gt|sketch|ref_<mode>/<id>.png          written by export_eval_assets.py
Output
    one appended row in results/<model>_<mode>.csv

Distribution metrics (FID/KID/FD-DINOv2) are computed per seed (579 outputs vs the 579 ground-truth
images, the prior-work protocol) and reported as mean and std over seeds.
"""

import argparse
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from src.metrics import MODE_METRICS, check_metrics
from src.metrics.distribution import DinoV2Features, InceptionFeatures, frechet_distance, kernel_distance
from src.metrics.edges import edge_f1
from src.metrics.files import index_by_id, index_outputs
from src.metrics.perceptual import LPIPS, pairwise_diversity
from src.metrics.pixel import ms_ssim, psnr, ssim
from src.metrics.reference import CLIPSimilarity, color_distance
from src.utils.image_io import load_png
from src.utils.seeding import SEEDS

Item = tuple[str, int, Path]  # (image_id, seed, path)


def collect_items(index: dict[str, dict[int, Path]], expected_ids: list[str], allow_partial: bool) -> list[Item]:
    missing = [i for i in expected_ids if i not in index]
    extra = sorted(set(index) - set(expected_ids))
    if extra:
        raise ValueError(f"{len(extra)} outputs have no ground-truth entry, e.g. {extra[:3]}")
    if missing and not allow_partial:
        raise ValueError(f"{len(missing)}/{len(expected_ids)} test ids have no output, e.g. {missing[:3]} (use --allow-partial)")
    seed_sets = {tuple(sorted(index[i])) for i in expected_ids if i in index}
    if len(seed_sets) > 1 and not allow_partial:
        raise ValueError(f"ids have different seed sets: {sorted(seed_sets)[:3]}")
    seeds = sorted(seed_sets)[0] if seed_sets else ()
    if seeds and list(seeds) != SEEDS and len(seeds) > 1:
        warnings.warn(f"seeds {list(seeds)} differ from SEEDS={SEEDS}")
    return [(i, s, index[i][s]) for i in expected_ids if i in index for s in sorted(index[i])]


def chunks(items: list, size: int):
    for k in range(0, len(items), size):
        yield items[k : k + size]


def load_rgb_uint8(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def paired_metrics(items: list[Item], gt: dict[str, Path], names: list[str], lpips_fn, batch_size: int) -> dict:
    fns = {"psnr": psnr, "ssim": ssim, "ms_ssim": ms_ssim, "lpips": lpips_fn}
    values = defaultdict(list)
    for batch in tqdm(list(chunks(items, batch_size)), desc="paired"):
        pred = torch.stack([load_png(p) for _, _, p in batch])
        target = torch.stack([load_png(gt[i]) for i, _, _ in batch])
        if pred.shape != target.shape:
            raise ValueError(f"output {tuple(pred.shape[1:])} vs ground truth {tuple(target.shape[1:])}: all models must output 256x256")
        for name in names:
            values[name].append(fns[name](pred, target).float().cpu())
    return {name: torch.cat(v).mean().item() for name, v in values.items()}


def diversity(index: dict[str, dict[int, Path]], ids: list[str], lpips_fn) -> dict:
    scores = []
    for image_id in tqdm(ids, desc="diversity"):
        seeds = sorted(index.get(image_id, {}))
        if len(seeds) < 2:
            continue
        samples = torch.stack([load_png(index[image_id][s]) for s in seeds])
        scores.append(pairwise_diversity(samples, lpips_fn))
    if not scores:
        warnings.warn("diversity needs >= 2 seeds per id; reporting NaN")
    return {"diversity": float(np.mean(scores)) if scores else float("nan")}


def edge_metrics(items: list[Item], sketches: dict[str, Path]) -> dict:
    prf = np.array([
        edge_f1(load_rgb_uint8(p), np.asarray(Image.open(sketches[i]).convert("L")))
        for i, _, p in tqdm(items, desc="edge_f1")
    ])
    return {"edge_precision": prf[:, 0].mean(), "edge_recall": prf[:, 1].mean(), "edge_f1": prf[:, 2].mean()}


def distribution_metrics(items: list[Item], real_files: list[Path], names: list[str], device, dino_model: str) -> dict:
    by_seed = defaultdict(list)
    for _, seed, path in items:
        by_seed[seed].append(path)
    out = {}
    extractors = []
    if "fid" in names or "kid" in names:
        extractors.append(("inception", InceptionFeatures(device=device)))
    if "fd_dinov2" in names:
        extractors.append(("dinov2", DinoV2Features(dino_model, device=device)))
    for kind, extract in extractors:
        real = extract(real_files)
        per_seed = defaultdict(list)
        for seed, files in tqdm(sorted(by_seed.items()), desc=f"{kind} features"):
            fake = extract(files)
            if kind == "inception":
                if "fid" in names:
                    per_seed["fid"].append(frechet_distance(fake, real))
                if "kid" in names:
                    per_seed["kid"].append(kernel_distance(fake, real))
            else:
                per_seed["fd_dinov2"].append(frechet_distance(fake, real))
        for name, vals in per_seed.items():
            out[name], out[f"{name}_std"] = float(np.mean(vals)), float(np.std(vals))
    return out


def reference_metrics(items: list[Item], refs: dict[str, Path], names: list[str], device, clip_model: str,
                      batch_size: int, use_mask: bool) -> dict:
    missing = sorted({i for i, _, _ in items} - set(refs))
    if missing:
        raise ValueError(f"{len(missing)} ids have no reference PNG, e.g. {missing[:3]}")
    out = {}
    if "color_dist" in names:
        out["color_dist"] = float(np.mean([
            color_distance(load_rgb_uint8(p), load_rgb_uint8(refs[i]), use_mask=use_mask)
            for i, _, p in tqdm(items, desc="color_dist")
        ]))
    if "clip_sim" in names:
        clip = CLIPSimilarity(clip_model, device=device)
        sims = [
            clip([load_rgb_uint8(p) for _, _, p in batch], [load_rgb_uint8(refs[i]) for i, _, _ in batch])
            for batch in tqdm(list(chunks(items, batch_size)), desc="clip_sim")
        ]
        out["clip_sim"] = torch.cat(sims).mean().item()
    return out


def append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if csv_path.exists():
        df = pd.concat([pd.read_csv(csv_path), df], ignore_index=True)
    df.to_csv(csv_path, index=False)


def evaluate(args) -> dict:
    metrics = args.metrics.split(",") if args.metrics else MODE_METRICS[args.mode]
    check_metrics(args.mode, metrics)
    pred_dir = args.pred_dir or Path("outputs") / args.model / args.mode
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    gt = index_by_id(args.eval_dir / "gt")
    index = index_outputs(pred_dir)
    ids = sorted(gt)
    items = collect_items(index, ids, args.allow_partial)
    present_ids = [i for i in ids if i in index]

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"), "model": args.model, "mode": args.mode,
        "n_ids": len(present_ids), "n_seeds": len({s for _, s, _ in items}),
    }
    needs_lpips = "lpips" in metrics or "diversity" in metrics
    lpips_fn = LPIPS(device=device) if needs_lpips else None

    paired = [m for m in metrics if m in ("psnr", "ssim", "ms_ssim", "lpips")]
    if paired:
        row.update(paired_metrics(items, gt, paired, lpips_fn, args.batch_size))
    if "diversity" in metrics:
        row.update(diversity(index, present_ids, lpips_fn))
    if "edge_f1" in metrics:
        row.update(edge_metrics(items, index_by_id(args.eval_dir / "sketch")))
    dist = [m for m in metrics if m in ("fid", "kid", "fd_dinov2")]
    if dist:
        row.update(distribution_metrics(items, [gt[i] for i in ids], dist, device, args.dino_model))
    ref = [m for m in metrics if m in ("color_dist", "clip_sim")]
    if ref:
        refs = index_by_id(args.eval_dir / f"ref_{args.mode}")
        row.update(reference_metrics(items, refs, ref, device, args.clip_model, args.batch_size, not args.no_color_mask))

    out = args.out or Path("results") / f"{args.model}_{args.mode}.csv"
    append_row(out, row)
    print(pd.Series(row).to_string())
    print(f"appended to {out}")
    return row


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="run name, e.g. gan, flow_steps8, controlnet_ip0.6")
    parser.add_argument("--mode", required=True, choices=sorted(MODE_METRICS))
    parser.add_argument("--pred-dir", type=Path, default=None, help="default: outputs/<model>/<mode>")
    parser.add_argument("--eval-dir", type=Path, default=Path("data/eval"))
    parser.add_argument("--metrics", default=None, help="comma list; default: every metric valid for --mode")
    parser.add_argument("--out", type=Path, default=None, help="default: results/<model>_<mode>.csv")
    parser.add_argument("--allow-partial", action="store_true", help="score a subset of test ids (never for reported numbers)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dino-model", default="facebook/dinov2-large")
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--no-color-mask", action="store_true", help="color histogram over the whole image")
    return parser.parse_args(argv)


if __name__ == "__main__":
    evaluate(parse_args())
