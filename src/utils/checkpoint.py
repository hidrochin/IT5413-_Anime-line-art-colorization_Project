"""Resumable checkpoints for every training script (Kaggle sessions stop after ~12 h).

    ckpt = CheckpointManager("/kaggle/working/checkpoints/flow", every_minutes=30,
                             init_from="/kaggle/input/<prev-session>/checkpoints/flow")
    state = ckpt.load_latest(model=model, ema=ema, optimizer=opt, scaler=scaler)
    step = state["step"] if state else 0
    sampler = EpochSampler(len(ds), seed=0); sampler.set_position(**state["data"]) if state else None
    for batch in loader:
        ...
        if ckpt.due():
            ckpt.save(step, model=model, ema=ema, optimizer=opt, scaler=scaler, data=sampler.position())
    ckpt.save(step, ...)  # always save once more at the end of the session

Each save writes <dir>/step_<step>.pt atomically (a session killed mid-save keeps the previous file),
including Python/NumPy/torch/CUDA RNG states, then deletes all but the newest `keep` files. Anything
with `state_dict()`/`load_state_dict()` is stored by its state dict; other values (dicts, numbers) are
stored as they are and returned by `load_latest`.
"""

import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Sampler

_NAME = re.compile(r"^step_(\d+)\.pt$")


def rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def list_checkpoints(folder: str | Path) -> list[tuple[int, Path]]:
    """[(step, path)] for step_*.pt in `folder`, oldest first."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    found = [(int(m.group(1)), p) for p in folder.iterdir() if (m := _NAME.match(p.name))]
    return sorted(found)


def unwrap(model: Any) -> Any:
    """Strip DataParallel/DDP/torch.compile wrappers so state-dict keys don't depend on how the model ran."""
    model = getattr(model, "_orig_mod", model)
    return getattr(model, "module", model)


class CheckpointManager:
    def __init__(self, folder: str | Path, every_minutes: float = 30, keep: int = 2,
                 init_from: str | Path | None = None):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.every_minutes = every_minutes
        self.keep = keep
        self.init_from = Path(init_from) if init_from else None
        self._last_save = time.time()

    def due(self) -> bool:
        """True once `every_minutes` have passed since the last save (or since construction)."""
        return (time.time() - self._last_save) / 60 >= self.every_minutes

    def latest(self) -> Path | None:
        """Newest checkpoint in `folder`, else in `init_from` (an earlier session's attached output)."""
        for folder in (self.folder, self.init_from):
            if folder is not None and (found := list_checkpoints(folder)):
                return found[-1][1]
        return None

    def save(self, step: int, **objects: Any) -> Path:
        payload = {"step": step, "rng": rng_state(), "objects": {}}
        for name, obj in objects.items():
            obj = unwrap(obj)
            payload["objects"][name] = obj.state_dict() if hasattr(obj, "state_dict") else obj
        path = self.folder / f"step_{step:08d}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        for _, old in list_checkpoints(self.folder)[: -self.keep]:
            old.unlink()
        self._last_save = time.time()
        print(f"saved checkpoint at step {step} -> {path}")
        return path

    def load_latest(self, restore_rng: bool = True, map_location: str = "cpu", **objects: Any) -> dict | None:
        """Restore `objects` in place from the newest checkpoint. Returns None if there is none, else
        {"step": int, "path": Path, <name>: value} for every stored entry that wasn't passed in."""
        path = self.latest()
        if path is None:
            return None
        payload = torch.load(path, map_location=map_location, weights_only=False)
        stored = payload["objects"]
        missing = set(objects) - set(stored)
        if missing:
            raise KeyError(f"{path} has no entries {sorted(missing)}; it holds {sorted(stored)}")
        for name, obj in objects.items():
            unwrap(obj).load_state_dict(stored[name])
        if restore_rng:
            set_rng_state(payload["rng"])
        print(f"resumed from {path} (step {payload['step']})")
        extra = {k: v for k, v in stored.items() if k not in objects}
        return {"step": payload["step"], "path": path, **extra}


class EpochSampler(Sampler):
    """Seeded permutation per epoch that can start mid-epoch, so a resumed run continues the same order.

    Store `position()` in the checkpoint and pass it back to `set_position` after resuming. Call
    `advance(n)` after each batch of n samples and `next_epoch()` when the loader is exhausted.
    """

    def __init__(self, n: int, seed: int):
        self.n, self.seed, self.epoch, self.start = n, seed, 0, 0

    def set_position(self, epoch: int, start: int) -> None:
        self.epoch, self.start = epoch, start

    def position(self) -> dict:
        return {"epoch": self.epoch, "start": self.start}

    def advance(self, n: int) -> None:
        self.start += n

    def next_epoch(self) -> None:
        self.epoch, self.start = self.epoch + 1, 0

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.n, generator=g).tolist()[self.start :])

    def __len__(self) -> int:
        return self.n - self.start
