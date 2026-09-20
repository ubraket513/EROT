"""Portable NumPy-based input and result serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .types import SolveResult


def load_array(path: str | Path) -> np.ndarray:
    source = Path(path)
    if source.suffix != ".npy":
        raise ValueError(f"expected a .npy array, received {source}")
    return np.load(source, allow_pickle=False)


def save_result(
    path: str | Path, result: SolveResult, *, metadata: dict[str, Any]
) -> Path:
    destination = Path(path)
    if destination.suffix != ".npz":
        raise ValueError("solver results must use the .npz extension")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        coupling=np.asarray(result.coupling),
        error=np.asarray(result.error),
        iterations=np.asarray(result.iterations),
        converged=np.asarray(result.converged),
        elapsed_seconds=np.asarray(result.elapsed_seconds),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return destination


def load_result(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.suffix != ".npz":
        raise ValueError(f"expected a .npz result, received {source}")
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "coupling",
            "error",
            "iterations",
            "converged",
            "elapsed_seconds",
            "metadata",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"result archive is missing: {', '.join(sorted(missing))}")
        return {
            "coupling": archive["coupling"],
            "error": float(archive["error"]),
            "iterations": int(archive["iterations"]),
            "converged": bool(archive["converged"]),
            "elapsed_seconds": float(archive["elapsed_seconds"]),
            "metadata": json.loads(str(archive["metadata"])),
        }
