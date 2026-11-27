"""Index generated PNGs named by the output contract: <image_id>_s<seed>.png."""

import re
from collections import defaultdict
from pathlib import Path

_NAME = re.compile(r"^(?P<id>.+)_s(?P<seed>\d+)$")


def parse_output_name(path: str | Path) -> tuple[str, int]:
    """'abc_s3.png' -> ('abc', 3). Files without a seed suffix (e.g. ground truth) count as seed 0."""
    stem = Path(path).stem
    m = _NAME.match(stem)
    if m is None:
        return stem, 0
    return m["id"], int(m["seed"])


def index_outputs(folder: str | Path) -> dict[str, dict[int, Path]]:
    """{image_id: {seed: path}} for every PNG in `folder`."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"output folder not found: {folder}")
    index: dict[str, dict[int, Path]] = defaultdict(dict)
    for p in sorted(folder.glob("*.png")):
        image_id, seed = parse_output_name(p)
        index[image_id][seed] = p
    return dict(index)


def index_by_id(folder: str | Path) -> dict[str, Path]:
    """{image_id: path} for a folder of <image_id>.png (ground truth, sketches, references)."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"folder not found: {folder}")
    return {p.stem: p for p in sorted(folder.glob("*.png"))}
