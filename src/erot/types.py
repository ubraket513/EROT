"""Public configuration and result types for EROT solvers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import jax

Precision = Literal["float32", "float64"]
DeviceSpec = str


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """Controls numerical precision, convergence, and device placement."""

    epsilon: float
    tolerance: float = 1e-8
    max_iterations: int = 50_000
    dtype: Precision = "float64"
    device: DeviceSpec = "auto"

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(v)
            for v in (self.epsilon, self.tolerance, self.max_iterations)
        ):
            raise ValueError("solver controls must be finite")
        if (
            self.max_iterations != int(self.max_iterations)
            or self.max_iterations > 2**31 - 1
        ):
            raise ValueError("max_iterations must be an integer fitting int32")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be greater than zero")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be greater than zero")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be greater than zero")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be 'float32' or 'float64'")


@dataclass(frozen=True, slots=True)
class SolveResult:
    """The coupling and convergence information returned by :func:`solve`."""

    coupling: jax.Array
    error: float
    iterations: int
    converged: bool
    elapsed_seconds: float
