"""Input validation and conversion at the public API boundary."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ._runtime import complex_dtype, real_dtype
from .types import Precision


def _finite(name: str, value: np.ndarray) -> None:
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")


def validate_classical(
    cost: ArrayLike,
    marginals: Sequence[ArrayLike],
    precision: Precision,
    device: jax.Device,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    if len(marginals) < 2:
        raise ValueError("classical problems require at least two marginals")

    host_cost = np.asarray(cost)
    if np.iscomplexobj(host_cost):
        raise ValueError("classical cost tensors must be real-valued")
    if host_cost.ndim != len(marginals):
        raise ValueError("cost rank must equal the number of marginals")
    _finite("cost", host_cost)

    host_marginals: list[np.ndarray] = []
    masses: list[float] = []
    for axis, marginal in enumerate(marginals):
        host = np.asarray(marginal)
        if np.iscomplexobj(host) or host.ndim != 1:
            raise ValueError(f"marginal {axis} must be a real one-dimensional array")
        if host.shape[0] != host_cost.shape[axis]:
            raise ValueError(
                f"marginal {axis} has length {host.shape[0]}, expected {host_cost.shape[axis]}"
            )
        _finite(f"marginal {axis}", host)
        if np.any(host < 0):
            raise ValueError(f"marginal {axis} must be nonnegative")
        mass = float(host.sum())
        if mass <= 0:
            raise ValueError(f"marginal {axis} must have positive mass")
        host_marginals.append(host)
        masses.append(mass)

    mass_tolerance = 1e-6 if precision == "float32" else 1e-10
    if not np.allclose(masses, masses[0], rtol=mass_tolerance, atol=mass_tolerance):
        raise ValueError("all marginals must have the same total mass")

    dtype = real_dtype(precision)
    cost_array = jax.device_put(jnp.asarray(host_cost, dtype=dtype), device)
    marginal_arrays = tuple(
        jax.device_put(jnp.asarray(marginal, dtype=dtype), device)
        for marginal in host_marginals
    )
    return cost_array, marginal_arrays


def validate_quantum(
    cost: ArrayLike,
    marginals: Sequence[ArrayLike],
    precision: Precision,
    device: jax.Device,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    if len(marginals) != 2:
        raise ValueError("quantum problems require exactly two density matrices")

    host_marginals = [np.asarray(marginal) for marginal in marginals]
    for index, marginal in enumerate(host_marginals):
        if marginal.ndim != 2 or marginal.shape[0] != marginal.shape[1]:
            raise ValueError(f"quantum marginal {index} must be a square matrix")
        _finite(f"quantum marginal {index}", marginal)

    n, m = host_marginals[0].shape[0], host_marginals[1].shape[0]
    host_cost = np.asarray(cost)
    if host_cost.shape != (n * m, n * m):
        raise ValueError(f"quantum cost must have shape {(n * m, n * m)}")
    _finite("cost", host_cost)

    tolerance = 1e-5 if precision == "float32" else 1e-9
    matrices = [("cost", host_cost)] + [
        (f"quantum marginal {index}", marginal)
        for index, marginal in enumerate(host_marginals)
    ]
    for name, matrix in matrices:
        if not np.allclose(matrix, matrix.conj().T, rtol=tolerance, atol=tolerance):
            raise ValueError(f"{name} must be Hermitian")

    traces: list[float] = []
    for index, marginal in enumerate(host_marginals):
        minimum_eigenvalue = float(np.linalg.eigvalsh(marginal).min())
        if minimum_eigenvalue < -tolerance:
            raise ValueError(f"quantum marginal {index} must be positive semidefinite")
        trace = np.trace(marginal)
        if abs(float(np.imag(trace))) > tolerance:
            raise ValueError(f"quantum marginal {index} must have a real trace")
        traces.append(float(np.real(trace)))

    if not np.allclose(traces, 1.0, rtol=tolerance, atol=tolerance):
        raise ValueError("quantum marginals must be trace-one density matrices")

    use_complex = np.iscomplexobj(host_cost) or any(
        np.iscomplexobj(marginal) for marginal in host_marginals
    )
    dtype = complex_dtype(precision) if use_complex else real_dtype(precision)
    cost_array = jax.device_put(jnp.asarray(host_cost, dtype=dtype), device)
    marginal_arrays = tuple(
        jax.device_put(jnp.asarray(marginal, dtype=dtype), device)
        for marginal in host_marginals
    )
    return cost_array, (marginal_arrays[0], marginal_arrays[1])
