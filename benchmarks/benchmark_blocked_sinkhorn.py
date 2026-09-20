"""Matched dense/blocked accuracy, synchronized timing and compiled memory evidence."""

from __future__ import annotations

import argparse
import json
import re
import resource
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from erot.geometry import PointCloudGeometry
from erot.geometry.plan import transport_objective
from erot.solvers.blocked_sinkhorn import solve_blocked_sinkhorn
from erot.solvers.sinkhorn import materialize_plan, solve_sinkhorn

if __package__:
    from .benchmark_solvers import (
        _case_digest,
        _memory_stats,
        _runtime_profile,
        _source_revision,
    )
    from .reporting import summarize_timings
else:
    from benchmark_solvers import (
        _case_digest,
        _memory_stats,
        _runtime_profile,
        _source_revision,
    )
    from reporting import summarize_timings


def compiled_memory(compiled, n, m):
    """Inspect executable shapes and compiler accounting, not measured GPU peak."""
    text = compiled.as_text() or ""
    shapes = {
        tuple(int(part) for part in shape.split(",") if part)
        for shape in re.findall(r"(?:bf|f|s|u|c)[0-9]+\[([0-9,]*)\]", text)
    }
    full = sorted(
        shape
        for shape in shapes
        if shape == (n * m,) or (len(shape) >= 2 and shape[:2] in ((n, m), (m, n)))
    )
    stats = compiled.memory_analysis()
    fields = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
        "alias_size_in_bytes",
    )
    values = {name: getattr(stats, name, None) for name in fields}
    total = (
        values[fields[0]] + values[fields[1]] + values[fields[2]] - values[fields[3]]
        if all(value is not None for value in values.values())
        else None
    )
    return {
        "full_matrix_shapes": [list(shape) for shape in full],
        "compiled_total_bytes": total,
        **values,
        "interpretation": "compiler buffer accounting; excludes allocator reservation, compilation and other executables",
    }


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def measure(function, args, repeats, n, m):
    start = time.perf_counter()
    compiled = function.lower(*args).compile()
    compilation = time.perf_counter() - start

    def once():
        start = time.perf_counter()
        result = jax.block_until_ready(compiled(*args))
        return result, time.perf_counter() - start

    result, first = once()
    once()
    samples = []
    for _ in range(repeats):
        result, elapsed = once()
        samples.append(elapsed)
    return result, {
        "compilation_seconds": compilation,
        "first_execution_seconds": first,
        "warm": summarize_timings(samples),
        "memory": compiled_memory(compiled, n, m),
    }


def run(
    *,
    n=257,
    m=383,
    features=3,
    block_size=32,
    epsilon=0.3,
    tolerance=1e-8,
    max_iterations=10000,
    dtype="float64",
    device="cpu",
    seed=42,
    repeats=7,
    compare_dense=False,
):
    if min(n, m, features, block_size, max_iterations, repeats) <= 0:
        raise ValueError("sizes, budgets and repeats must be positive")
    if not np.isfinite([epsilon, tolerance]).all() or min(epsilon, tolerance) <= 0:
        raise ValueError("epsilon and tolerance must be finite and positive")
    jax.config.update("jax_enable_x64", dtype == "float64")
    target = jax.devices(device)[
        0
    ]  # Explicit gpu selection fails rather than falling back.
    rng = np.random.default_rng(seed)
    x, y = (
        rng.normal(size=(n, features)).astype(dtype),
        rng.normal(size=(m, features)).astype(dtype),
    )
    a, b = np.full(n, 1 / n, dtype=dtype), np.full(m, 1 / m, dtype=dtype)
    geometry = PointCloudGeometry(jax.device_put(x, target), jax.device_put(y, target))
    marginals = (jax.device_put(a, target), jax.device_put(b, target))
    controls = (
        jax.device_put(jnp.asarray(epsilon, dtype), target),
        jax.device_put(jnp.asarray(tolerance, dtype), target),
        jax.device_put(jnp.asarray(max_iterations, jnp.int32), target),
    )
    args = (geometry, marginals, *controls)
    result, blocked = measure(
        jax.jit(partial(solve_blocked_sinkhorn, block_size=block_size)),
        args,
        repeats,
        n,
        m,
    )
    state, diagnostics = result
    objective = float(
        jax.jit(partial(transport_objective, block_size=block_size))(
            geometry, state.potentials, controls[0]
        )
    )
    blocked.update(
        status=int(diagnostics.status),
        residual=_finite_or_none(diagnostics.error),
        iterations=int(diagnostics.iterations),
        objective=_finite_or_none(objective),
        device_memory=_memory_stats([target]),
    )
    report = {
        "schema": "erot.blocked-benchmark.v1",
        "n": n,
        "m": m,
        "features": features,
        "block_size": block_size,
        "epsilon": epsilon,
        "tolerance": tolerance,
        "max_iterations": max_iterations,
        "dtype": dtype,
        "seed": seed,
        "input_digest": _case_digest(x, y, a, b),
        "source_revision": _source_revision(),
        "runtime": _runtime_profile(),
        "device": str(target),
        "device_kind": target.device_kind,
        "output_contract": "potentials and diagnostics; no dense plan returned",
        "objective_convention": "sum(C*pi)+epsilon*sum(pi*(log(pi)-1))",
        "blocked": blocked,
        "dense": None,
        "memory_note": "device peaks are cumulative process allocator observations; compiler accounting is separate",
    }
    if compare_dense:
        cost = jnp.sum((geometry.x[:, None, :] - geometry.y[None, :, :]) ** 2, axis=-1)
        dense_result, dense = measure(
            jax.jit(solve_sinkhorn), (cost, marginals, *controls), repeats, n, m
        )
        dense_state, dense_diag = dense_result
        plan = materialize_plan(cost, dense_state.potentials, controls[0])
        dense_objective = float(
            jnp.sum(
                cost * plan + controls[0] * (jax.scipy.special.xlogy(plan, plan) - plan)
            )
        )
        dense.update(
            status=int(dense_diag.status),
            residual=_finite_or_none(dense_diag.error),
            iterations=int(dense_diag.iterations),
            objective=_finite_or_none(dense_objective),
            device_memory=_memory_stats([target]),
        )
        report["dense"] = dense
        report["objective_difference"] = _finite_or_none(
            abs(objective - dense_objective)
        )
        report["matched_accuracy"] = bool(
            diagnostics.status == 0
            and dense_diag.status == 0
            and abs(objective - dense_objective)
            <= max(100 * tolerance, 1e-6 if dtype == "float32" else 1e-10)
        )
    report["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (
        ("n", 257),
        ("m", 383),
        ("features", 3),
        ("block-size", 32),
        ("max-iterations", 10000),
        ("repeats", 7),
        ("seed", 42),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--compare-dense", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    output = args.pop("output")
    report = run(**args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "status": report["blocked"]["status"],
                "compiled_bytes": report["blocked"]["memory"]["compiled_total_bytes"],
            }
        )
    )
    return (
        0
        if report["blocked"]["status"] == 0
        and report["blocked"]["objective"] is not None
        and report.get("matched_accuracy", True)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
