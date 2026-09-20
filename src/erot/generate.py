"""Small dense problem generators used by the CLI and examples."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def cost_tensor(
    n: int,
    n_marginals: int = 2,
    *,
    kind: str = "euclidean",
    dtype: str = "float64",
    diagonal_penalty: float = 1e6,
) -> np.ndarray:
    if n <= 0 or n_marginals < 2:
        raise ValueError("n must be positive and n_marginals must be at least two")
    if dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be 'float32' or 'float64'")
    if diagonal_penalty <= 0 or not np.isfinite(diagonal_penalty):
        raise ValueError("diagonal_penalty must be positive and finite")

    coordinates = []
    for axis in range(n_marginals):
        shape = [1] * n_marginals
        shape[axis] = n
        coordinates.append(np.arange(n, dtype=dtype).reshape(shape))

    result = np.zeros((n,) * n_marginals, dtype=dtype)
    for first in range(n_marginals):
        for second in range(first + 1, n_marginals):
            distance = np.abs(coordinates[first] - coordinates[second])
            if kind == "euclidean":
                result = result + distance**2
            elif kind == "weak-coulomb":
                result = result + 1.0 / (distance + 1.0)
            elif kind == "strong-coulomb":
                result = result + np.where(
                    distance > 0, 1.0 / np.maximum(distance, 1.0), diagonal_penalty
                )
            else:
                raise ValueError(
                    "kind must be 'euclidean', 'weak-coulomb', or 'strong-coulomb'"
                )
    return result


def gaussian_marginal(
    n: int,
    *,
    locs: Sequence[float] = (0.0,),
    scales: Sequence[float] = (1.0,),
    lower: float = -5.0,
    upper: float = 5.0,
    dtype: str = "float64",
) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive")
    if len(locs) == 0 or len(locs) != len(scales):
        raise ValueError("locs and scales must be nonempty and have equal length")
    if any(scale <= 0 for scale in scales):
        raise ValueError("all scales must be positive")
    if not lower < upper:
        raise ValueError("lower must be less than upper")

    x = np.linspace(lower, upper, n, dtype=dtype)
    result = np.zeros(n, dtype=dtype)
    normalizer = np.sqrt(2.0 * np.pi)
    for location, scale in zip(locs, scales, strict=True):
        result += np.exp(-0.5 * ((x - location) / scale) ** 2) / (scale * normalizer)
    result /= len(locs)
    return result / result.sum()
