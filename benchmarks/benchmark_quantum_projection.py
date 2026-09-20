"""Isolated dense Hermitian eigensolver and PSD projection measurements."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

if __package__:
    from .benchmark_solvers import _memory_stats, _runtime_profile, _source_revision
    from .reporting import summarize_timings
else:
    from benchmark_solvers import _memory_stats, _runtime_profile, _source_revision
    from reporting import summarize_timings


def memory_budget(n, m, dtype):
    """Count dense state arrays; workspace and measured peak remain unknown."""
    if n <= 0 or m <= 0:
        raise ValueError("subsystem dimensions must be positive")
    if dtype not in ("complex64", "complex128"):
        raise ValueError("expected complex64 or complex128")
    size = np.dtype(dtype).itemsize * (n * m) ** 2
    return {
        "matrix_bytes": size,
        "persistent_solver_arrays_bytes": 5 * size,
        "persistent_arrays": [
            "cost",
            "coupling",
            "correction_1",
            "correction_2",
            "correction_3",
        ],
        "projection_intermediates_estimate_bytes": 3 * size,
        "eigensolver_workspace_bytes": None,
        "measured_peak_bytes": None,
        "interpretation": "array accounting only; not an upper bound on peak allocation",
    }


def project_psd(matrix):
    hermitian = (matrix + matrix.conj().T) / 2
    values, vectors = jnp.linalg.eigh(hermitian)
    projected = (vectors * jnp.maximum(values, 0)[None, :]) @ vectors.conj().T
    return (projected + projected.conj().T) / 2


def _measure(function, matrix):
    def once():
        start = time.perf_counter()
        result = jax.block_until_ready(function(matrix))
        return result, time.perf_counter() - start

    result, first = once()
    del result
    result, _ = once()
    del result
    samples = []
    for index in range(7):
        result, seconds = once()
        samples.append(seconds)
        if index < 6:
            del result
    return result, {"first_call_seconds": first, **summarize_timings(samples)}


def benchmark_projection(dimension, dtype, device, seed):
    if dimension <= 0:
        raise ValueError("matrix dimension must be positive")
    if dtype not in ("complex64", "complex128"):
        raise ValueError("expected complex64 or complex128")
    # Preserve requested complex precision before creating device arrays.
    jax.config.update("jax_enable_x64", True)
    target = jax.devices(device)[0]
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    host = ((z + z.conj().T) / (2 * np.sqrt(dimension))).astype(dtype)
    matrix = jax.device_put(host, target)
    matrix.block_until_ready()
    eigenpair, eigen_timing = _measure(jax.jit(jnp.linalg.eigh), matrix)
    del eigenpair
    eigen_memory = _memory_stats([target])
    projected, projection_timing = _measure(jax.jit(project_psd), matrix)
    projection_memory = _memory_stats([target])
    actual = np.asarray(projected)
    # Untimed double-precision NumPy reference, also for complex64 execution.
    values, vectors = np.linalg.eigh(host.astype(np.complex128))
    reference = (vectors * np.maximum(values, 0)) @ vectors.conj().T
    scale = max(1.0, float(np.linalg.norm(reference)))
    error = float(np.linalg.norm(actual - reference) / scale)
    hermiticity = float(np.linalg.norm(actual - actual.conj().T) / scale)
    minimum = float(np.linalg.eigvalsh(actual).min())
    tolerance = 2e-5 if dtype == "complex64" else 1e-11
    valid = bool(
        np.isfinite(actual).all()
        and error <= tolerance
        and hermiticity <= tolerance
        and minimum >= -tolerance
    )
    return {
        "schema_version": 1,
        "experiment": "dense-quantum-psd-projection",
        "dimension": dimension,
        "dtype": str(matrix.dtype),
        "seed": seed,
        "validated": valid,
        "validation_scope": "full independent NumPy complex128 projection",
        "reference_relative_error": error,
        "hermiticity_relative_error": hermiticity,
        "minimum_eigenvalue": minimum,
        "tolerance": tolerance,
        "eigh": eigen_timing,
        "projection": projection_timing,
        "memory_after_eigh": eigen_memory,
        "memory_after_projection": projection_memory,
        "memory_scope": "process cumulative high water; both operations share one process",
        "devices": [
            {"platform": target.platform, "kind": target.device_kind, "id": target.id}
        ],
        "runtime_profile": _runtime_profile(),
        "source_revision": _source_revision(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=16)
    parser.add_argument(
        "--dtype", choices=("complex64", "complex128"), default="complex128"
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.dimension <= 0:
        parser.error("dimension must be positive")
    report = benchmark_projection(args.dimension, args.dtype, args.device, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return int(not report["validated"])


if __name__ == "__main__":
    raise SystemExit(main())
