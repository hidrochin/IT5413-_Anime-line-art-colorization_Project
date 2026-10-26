"""Resolve "package.module:attr" strings, so scripts can take the dataset factory as a flag."""

import importlib
from typing import Any

DEFAULT_DATASET = "src.data:build_dataset"


_HINT = ("The shared Dataset (src/data, owner: Cường) must expose build_dataset(split, mode) "
         "returning dicts with id/sketch/target/reference.")


def load_object(spec: str) -> Any:
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"expected 'module:attr', got {spec!r}")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"cannot import {module_name!r} for {spec!r}. {_HINT}") from e
    if not hasattr(module, attr):
        raise AttributeError(f"{module_name!r} has no {attr!r} (from {spec!r}). {_HINT}")
    return getattr(module, attr)
