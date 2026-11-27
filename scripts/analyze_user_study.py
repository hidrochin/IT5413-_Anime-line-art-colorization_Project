"""Analyze the user study: mean rank and first-place rate per model, plus a Friedman test.

    python -m scripts.analyze_user_study --study-dir results/user_study

Reads <study-dir>/key.csv and every <study-dir>/responses/*.csv (rater,case,criterion,option,rank),
writes results/user_study_summary.csv.
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import friedmanchisquare


def summarize(responses: pd.DataFrame, key: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = responses.merge(key, on=["case", "option"], how="left", validate="many_to_one")
    if df["model"].isna().any():
        raise ValueError("responses reference case/option pairs that are not in key.csv")
    dup = df.duplicated(["rater", "case", "criterion", "option"], keep="last")
    df = df[~dup]  # a rater who submitted twice: keep the later answers

    summary = (
        df.groupby(["criterion", "model"])["rank"]
        .agg(mean_rank="mean", sem="sem", n="count", first_place=lambda r: (r == 1).mean())
        .reset_index()
        .sort_values(["criterion", "mean_rank"])
    )
    tests = {}
    for criterion, g in df.groupby("criterion"):
        wide = g.pivot_table(index=["rater", "case"], columns="model", values="rank").dropna()
        if wide.shape[1] >= 3 and len(wide) >= 2:
            stat, p = friedmanchisquare(*[wide[c] for c in wide.columns])
            tests[criterion] = {"friedman_chi2": stat, "p_value": p, "n_blocks": len(wide)}
    return summary, tests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study-dir", type=Path, default=Path("results/user_study"))
    parser.add_argument("--out", type=Path, default=Path("results/user_study_summary.csv"))
    args = parser.parse_args()

    files = sorted((args.study_dir / "responses").glob("*.csv"))
    if not files:
        raise SystemExit(f"no response CSVs in {args.study_dir / 'responses'}")
    responses = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    key = pd.read_csv(args.study_dir / "key.csv")
    summary, tests = summarize(responses, key)

    summary["friedman_p"] = summary["criterion"].map(lambda c: tests.get(c, {}).get("p_value"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(f"{responses['rater'].nunique()} raters, {responses['case'].nunique()} cases")
    print(summary.to_string(index=False))
    for criterion, t in tests.items():
        print(f"{criterion}: Friedman chi2={t['friedman_chi2']:.2f}, p={t['p_value']:.3g} ({t['n_blocks']} rater-case blocks)")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
