"""Cache SD-VAE latents of sketches and targets for Model B (encode once, train for days).

    python -m scripts.cache_latents --splits train val test --out data/latents

Writes <out>/<split>.pt = {"ids": [str], "sketch": (N,4,32,32) fp16, "target": (N,4,32,32) fp16,
"vae": model id, "scaling_factor": float}. Latents are posterior means already multiplied by the
scaling factor (see src/utils/vae.py). About 0.5 GB for train. References are *not* cached: training
references are re-warped every step, so their DINOv2 tokens are computed on the fly.

Upload <out> as a private Kaggle Dataset so other sessions can attach it.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.registry import DEFAULT_DATASET, load_object
from src.utils.vae import VAE_ID, encode, load_vae


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--vae", default=VAE_ID)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("data/latents"))
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    vae = load_vae(args.vae, args.device, dtype)
    build_dataset = load_object(args.dataset)
    args.out.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        # self_ref mode: sketch and target are the un-augmented pair (the reference is ignored here).
        dataset = build_dataset(split, "self_ref")
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        ids, sketch, target = [], [], []
        for batch in tqdm(loader, desc=split):
            ids += list(batch["id"])
            sketch.append(encode(vae, batch["sketch"]).half().cpu())
            target.append(encode(vae, batch["target"]).half().cpu())
        path = args.out / f"{split}.pt"
        torch.save({"ids": ids, "sketch": torch.cat(sketch), "target": torch.cat(target), "vae": args.vae,
                    "scaling_factor": float(vae.config.scaling_factor)}, path)
        print(f"{split}: {len(ids)} pairs -> {path}")


if __name__ == "__main__":
    main()
