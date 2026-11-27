"""Training-curve logger: one CSV row per call, appended across resumed sessions, echoed to stdout.

    log = CSVLogger(out_dir / "log.csv")
    log.log(step, loss=0.12, lr=1e-4)

Columns are fixed by the first row ever written to the file (kept when a session resumes). Unknown
keys in later calls raise, so a typo doesn't silently start a new column.
"""

import csv
import math
import time
from pathlib import Path


class CSVLogger:
    def __init__(self, path: str | Path, echo: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self.start = time.time()
        self.columns: list[str] | None = None
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open(newline="") as f:
                self.columns = next(csv.reader(f))

    def log(self, step: int, **metrics: float) -> None:
        row = {"step": step, **metrics, "elapsed_min": round((time.time() - self.start) / 60, 2)}
        if self.columns is None:
            self.columns = list(row)
            with self.path.open("w", newline="") as f:
                csv.writer(f).writerow(self.columns)
        unknown = set(row) - set(self.columns)
        if unknown:
            raise KeyError(f"{sorted(unknown)} not in the columns of {self.path}: {self.columns}")
        with self.path.open("a", newline="") as f:
            csv.writer(f).writerow([row.get(c, "") for c in self.columns])
        if self.echo:
            parts = [f"{k} {_fmt(v)}" for k, v in metrics.items()]
            print(f"step {step}: " + ", ".join(parts) + f" ({row['elapsed_min']:.1f} min)")


def _fmt(v) -> str:
    if isinstance(v, float) and math.isfinite(v):
        return f"{v:.4g}"
    return str(v)
