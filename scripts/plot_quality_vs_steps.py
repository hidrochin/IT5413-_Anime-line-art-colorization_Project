"""Experiment 3 (RQ2): quality vs sampling cost, self_ref mode.

    python -m scripts.plot_quality_vs_steps --series "flow_steps*" "controlnet_steps*" --points gan

x = seconds per image from results/speed.csv (written by sample.py; joined on run name), or the
step count parsed from the run name if speed is missing. One panel per --metrics entry.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from scripts.results_io import load_results, match, sweep_value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--series", nargs="+", required=True)
    parser.add_argument("--points", nargs="*", default=["gan"], help="single-step runs drawn as one marker")
    parser.add_argument("--metrics", nargs="+", default=["fd_dinov2", "lpips"])
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results/figures/quality_vs_steps.png"))
    args = parser.parse_args()

    df = load_results(args.results_dir, "self_ref")
    speed_path = args.results_dir / "speed.csv"
    if speed_path.exists():
        speed = pd.read_csv(speed_path).groupby("model", as_index=False).last()[["model", "sec_per_image"]]
        df = df.merge(speed, on="model", how="left")
    else:
        df["sec_per_image"] = float("nan")
    use_time = df["sec_per_image"].notna().any()

    fig, axes = plt.subplots(1, len(args.metrics), figsize=(5.5 * len(args.metrics), 4.2), squeeze=False)
    for ax, metric in zip(axes[0], args.metrics):
        for pattern in args.series:
            runs = match(df, pattern).dropna(subset=[metric])
            if runs.empty:
                continue
            runs = runs.assign(steps=runs["model"].map(sweep_value)).sort_values("steps")
            x = runs["sec_per_image"] if use_time else runs["steps"]
            ax.plot(x, runs[metric], marker="o", label=pattern.rstrip("*_"))
            for xi, yi, s in zip(x, runs[metric], runs["steps"]):
                ax.annotate(f"{s:g}", (xi, yi), textcoords="offset points", xytext=(3, 3), fontsize=7)
        for name in args.points:
            row = df[df["model"] == name].dropna(subset=[metric])
            if not row.empty:
                x = row["sec_per_image"] if use_time else pd.Series([1.0])
                ax.scatter(x, row[metric], marker="*", s=150, zorder=3, label=name)
        ax.set_xscale("log")
        ax.set_xlabel("seconds / image (T4)" if use_time else "sampling steps")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3, which="both")
        ax.legend()
    fig.suptitle("Quality vs sampling cost (self-reference)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
