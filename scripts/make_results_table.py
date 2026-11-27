"""Experiment 1 table: latest results per model for one mode, next to the prior-work numbers.

    python -m scripts.make_results_table --mode self_ref --models gan flow controlnet controlnet_lora
"""

import argparse
from pathlib import Path

import pandas as pd

from scripts.results_io import load_results

# AnimeDiffusion (Cao et al., TVCG 2024), Table 1, official 579-image test set. PSNR / MS-SSIM in
# self-reference mode; FID with references shuffled across the test set (our shuffled_ref mode).
PRIOR_WORK = pd.DataFrame([
    {"model": "SCFT (CVPR 2020)", "psnr": 23.89, "ms_ssim": 0.922, "fid": 57.19},
    {"model": "Li et al. 2022", "psnr": 18.63, "ms_ssim": 0.821, "fid": 49.33},
    {"model": "Cao et al. 2022", "psnr": 19.77, "ms_ssim": 0.839, "fid": 46.39},
    {"model": "AnimeDiffusion (TVCG 2024)", "psnr": 25.47, "ms_ssim": 0.960, "fid": 44.19},
])

COLUMNS = {
    "self_ref": ["psnr", "ssim", "ms_ssim", "lpips", "fid", "kid", "fd_dinov2", "diversity", "edge_f1"],
    "cross_ref": ["fid", "kid", "fd_dinov2", "color_dist", "clip_sim", "diversity", "edge_f1"],
    "shuffled_ref": ["fid", "kid", "fd_dinov2", "diversity", "edge_f1"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=sorted(COLUMNS))
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--no-prior", action="store_true")
    args = parser.parse_args()

    df = load_results(args.results_dir, args.mode)
    missing = [m for m in args.models if m not in set(df["model"])]
    if missing:
        raise SystemExit(f"no {args.mode} results for {missing}")
    ours = df.set_index("model").loc[args.models].reset_index()
    cols = ["model"] + [c for c in COLUMNS[args.mode] if c in ours.columns]
    table = ours[cols]
    # Prior work: PSNR / MS-SSIM are self_ref numbers, FID is a shuffled_ref number.
    if not args.no_prior and args.mode in ("self_ref", "shuffled_ref"):
        prior_cols = ["psnr", "ms_ssim"] if args.mode == "self_ref" else ["fid"]
        prior = PRIOR_WORK[["model"] + prior_cols].assign(model=lambda d: d["model"] + " †")
        table = pd.concat([prior, table], ignore_index=True)[cols]

    out = args.results_dir / f"table_{args.mode}"
    table.to_csv(out.with_suffix(".csv"), index=False)
    md = table.to_markdown(index=False, floatfmt=".4g")
    if not args.no_prior and args.mode in ("self_ref", "shuffled_ref"):
        md += "\n\n† reported in AnimeDiffusion (Cao et al., TVCG 2024), Table 1."
    out.with_suffix(".md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
