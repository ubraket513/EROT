"""Unified public API for classical and quantum EROT."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Literal

import jax
import jax.numpy as jnp

from . import classical, quantum
from ._runtime import enable_requested_precision, resolve_device
from .types import SolverConfig, SolveResult
from .validation import ArrayLike, validate_classical, validate_quantum

Problem = Literal["classical", "quantum"]
Regularizer = Literal["shannon", "quadratic"]
Method = Literal["sinkhorn", "cyclic"]


def solve(
    cost: ArrayLike,
    marginals: Sequence[ArrayLike],
    *,
    problem: Problem,
    regularizer: Regularizer,
    method: Method,
    config: SolverConfig,
) -> SolveResult:
    """Solve a supported entropy-regularized optimal transport problem.

    Compilation time is included in ``elapsed_seconds`` on the first call for a
    new shape. Benchmarking utilities report compilation and warm execution
    separately.
    """

    enable_requested_precision(config.dtype)
    device = resolve_device(config.device)

    start = time.perf_counter()
    if problem == "classical":
        cost_array, marginal_arrays = validate_classical(
            cost, marginals, config.dtype, device
        )
        scalar_dtype = cost_array.dtype
        epsilon = jax.device_put(
            jnp.asarray(config.epsilon, dtype=scalar_dtype), device
        )
        tolerance = jax.device_put(
            jnp.asarray(config.tolerance, dtype=scalar_dtype), device
        )
        max_iterations = jax.device_put(
            jnp.asarray(config.max_iterations, dtype=jnp.int32), device
        )

        if regularizer == "shannon" and method == "sinkhorn":
            coupling, error, iterations = classical.shannon_sinkhorn(
                cost_array,
                marginal_arrays,
                epsilon,
                tolerance,
                max_iterations,
            )
        elif regularizer == "quadratic" and method == "cyclic":
            if len(marginal_arrays) != 2:
                raise ValueError(
                    "quadratic cyclic projection currently supports exactly two marginals"
                )
            coupling, error, iterations = classical.quadratic_cyclic_projection(
                cost_array,
                marginal_arrays[0],
                marginal_arrays[1],
                epsilon,
                tolerance,
                max_iterations,
            )
        else:
            raise ValueError(
                "supported classical combinations are shannon/sinkhorn and quadratic/cyclic"
            )
    elif problem == "quantum":
        if regularizer != "quadratic" or method != "cyclic":
            raise ValueError("the stable quantum solver supports only quadratic/cyclic")
        cost_array, marginal_arrays = validate_quantum(
            cost, marginals, config.dtype, device
        )
        scalar_dtype = cost_array.real.dtype
        epsilon = jax.device_put(
            jnp.asarray(config.epsilon, dtype=scalar_dtype), device
        )
        tolerance = jax.device_put(
            jnp.asarray(config.tolerance, dtype=scalar_dtype), device
        )
        max_iterations = jax.device_put(
            jnp.asarray(config.max_iterations, dtype=jnp.int32), device
        )
        coupling, error, iterations = quantum.quadratic_cyclic_projection(
            cost_array,
            marginal_arrays[0],
            marginal_arrays[1],
            epsilon,
            tolerance,
            max_iterations,
        )
    else:
        raise ValueError("problem must be 'classical' or 'quantum'")

    coupling.block_until_ready()
    elapsed_seconds = time.perf_counter() - start
    error_value = float(error)
    iteration_count = int(iterations)
    return SolveResult(
        coupling=coupling,
        error=error_value,
        iterations=iteration_count,
        converged=error_value <= config.tolerance,
        elapsed_seconds=elapsed_seconds,
    )
