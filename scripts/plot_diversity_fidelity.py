"""Experiment 2 (RQ1): diversity (pairwise LPIPS) vs fidelity (LPIPS to ground truth), self_ref mode.

    python -m scripts.plot_diversity_fidelity --series "gan" "flow_w*" "controlnet_ip*"

Each --series is a glob over run names and becomes one curve, ordered by the run's trailing number
(the swept guidance weight w or IP-Adapter scale λ). A single run (e.g. gan) is drawn as one point.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.results_io import load_results, match, sweep_value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--series", nargs="+", required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results/figures/diversity_fidelity.png"))
    args = parser.parse_args()

    df = load_results(args.results_dir, "self_ref")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for pattern in args.series:
        runs = match(df, pattern).dropna(subset=["diversity", "lpips"])
        if runs.empty:
            print(f"warning: no self_ref runs with diversity+lpips match {pattern!r}")
            continue
        runs = runs.assign(v=runs["model"].map(sweep_value)).sort_values("v")
        label = pattern.rstrip("*_")
        ax.plot(runs["diversity"], runs["lpips"], marker="o", label=label, linestyle="-" if len(runs) > 1 else "none")
        if len(runs) > 1:
            for _, r in runs.iterrows():
                ax.annotate(f"{r['v']:g}", (r["diversity"], r["lpips"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("diversity: pairwise LPIPS over 5 seeds (higher = more varied)")
    ax.set_ylabel("LPIPS to ground truth (lower = more faithful)")
    ax.set_title("Diversity vs fidelity (self-reference)")
    ax.grid(alpha=0.3)
    ax.legend()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
