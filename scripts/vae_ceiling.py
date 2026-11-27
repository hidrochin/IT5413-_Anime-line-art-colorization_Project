"""VAE reconstruction ceiling: encode -> decode the ground truth, so evaluate.py can score the best a
latent model (B, C) could possibly do at 256 px.

    python -m scripts.vae_ceiling --split test
    python -m scripts.evaluate --model vae_ceiling --mode self_ref

Writes outputs/vae_ceiling/self_ref/<id>_s0.png (one seed: the posterior mean is deterministic, so
diversity is reported as NaN for this run).
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.image_io import save_png
from src.utils.registry import DEFAULT_DATASET, load_object
from src.utils.vae import VAE_ID, decode, encode, load_vae


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test")
    parser.add_argument("--vae", default=VAE_ID)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", type=Path, default=Path("outputs/vae_ceiling/self_ref"))
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    vae = load_vae(args.vae, args.device, dtype)
    dataset = load_object(args.dataset)(args.split, "self_ref")
    for batch in tqdm(DataLoader(dataset, batch_size=args.batch_size), desc="vae ceiling"):
        recon = decode(vae, encode(vae, batch["target"])).cpu()
        for image_id, img in zip(batch["id"], recon):
            save_png(img, args.out / f"{image_id}_s0.png")
    print(f"wrote {len(dataset)} reconstructions to {args.out}")


if __name__ == "__main__":
    main()
