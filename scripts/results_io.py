"""Shared helpers for the plotting/table scripts: latest result row per run, sweep-value parsing."""

import fnmatch
import re
from pathlib import Path

import pandas as pd


def load_results(results_dir: Path, mode: str) -> pd.DataFrame:
    """Latest row per model from results/*_<mode>.csv."""
    frames = [pd.read_csv(p) for p in sorted(results_dir.glob(f"*_{mode}.csv"))]
    if not frames:
        raise FileNotFoundError(f"no results/*_{mode}.csv in {results_dir}; run scripts.evaluate first")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["mode"] == mode].sort_values("timestamp")
    return df.groupby("model", as_index=False).last()


def match(df: pd.DataFrame, pattern: str) -> pd.DataFrame:
    return df[[fnmatch.fnmatchcase(m, pattern) for m in df["model"]]]


def sweep_value(model: str) -> float:
    """'flow_steps8' -> 8.0, 'controlnet_ip0.6' -> 0.6; NaN if there is no trailing number."""
    m = re.search(r"(\d+(?:\.\d+)?)$", model)
    return float(m.group(1)) if m else float("nan")
