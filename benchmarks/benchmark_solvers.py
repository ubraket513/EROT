"""Reproducible compile/warm benchmarks for EROT's stable solvers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import jax
import numpy as np

import erot

if __package__:
    from .reporting import assert_comparable, summarize_timings, validate_record
else:
    from reporting import assert_comparable, summarize_timings, validate_record


def _memory_stats(devices) -> list[dict[str, object]]:
    records = []
    for device in devices:
        stats = device.memory_stats() or {}
        records.append(
            {
                "device": str(device),
                **{
                    key: int(stats[key]) if key in stats else None
                    for key in (
                        "bytes_in_use",
                        "peak_bytes_in_use",
                        "bytes_reserved",
                        "bytes_limit",
                    )
                },
            }
        )
    return records


def _time_call(
    callable_: Callable[[], erot.SolveResult],
) -> tuple[erot.SolveResult, float]:
    start = time.perf_counter()
    result = callable_()
    result.coupling.block_until_ready()
    return result, time.perf_counter() - start


def _case_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _runtime_profile() -> dict[str, object]:
    versions = {"python": platform.python_version()}
    for name in ("jax", "jaxlib", "numpy", "jax-cuda12-plugin", "jax-cuda13-plugin"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    versions["environment"] = {
        key: os.environ.get(key)
        for key in (
            "JAX_PLATFORMS",
            "CUDA_VISIBLE_DEVICES",
            "XLA_FLAGS",
            "XLA_PYTHON_CLIENT_PREALLOCATE",
            "XLA_PYTHON_CLIENT_MEM_FRACTION",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
        )
    }
    return versions


def _source_revision() -> str | None:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parents[1]),
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _source_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    paths = list((root / "src/erot").rglob("*.py"))
    paths += list((root / "benchmarks").glob("*.py"))
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _measure(run, problem, size, cost, marginals, config):
    def objective(result):
        coupling = np.asarray(result.coupling)
        if not result.converged or not np.isfinite(coupling).all():
            raise ValueError("benchmark requires a finite converged plan")
        if problem == "classical-shannon":
            positive = coupling > 0
            entropy = np.sum(coupling[positive] * (np.log(coupling[positive]) - 1))
            return float(np.sum(cost * coupling) + config.epsilon * entropy)
        return float(
            np.trace(cost @ coupling).real
            + 0.5 * config.epsilon * np.vdot(coupling, coupling).real
        )

    first, first_seconds = _time_call(run)
    objective(first)
    first_iterations = first.iterations
    del first
    warmup, _ = _time_call(run)
    objective(warmup)
    del warmup
    samples = []
    for index in range(7):
        result, seconds = _time_call(run)
        value = objective(result)
        if not np.isfinite(value) or not np.isfinite(result.error):
            raise ValueError("nonfinite objective or residual in timed solve")
        samples.append(seconds)
        if index < 6:
            del result
    timing = summarize_timings(samples)
    devices = sorted(
        result.coupling.devices(), key=lambda item: (item.process_index, item.id)
    )
    record = {
        "schema_version": 2,
        "problem": problem,
        "size": size,
        "case_digest": _case_digest(cost, *marginals),
        "objective_convention": "coupling-entropy-minus-mass"
        if problem == "classical-shannon"
        else "quantum-quadratic-half-norm",
        "dtype": str(result.coupling.dtype),
        "geometry": "dense-squared-euclidean"
        if problem == "classical-shannon"
        else "dense-hermitian-cost",
        "shape": list(cost.shape),
        "epsilon": config.epsilon,
        "tolerance": config.tolerance,
        "max_iterations": config.max_iterations,
        "output_policy": "dense-coupling",
        "hardware_profile": {
            "host": platform.node(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "cpu_affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            "devices": [
                {
                    "id": item.id,
                    "process_index": item.process_index,
                    "platform": item.platform,
                    "kind": item.device_kind,
                }
                for item in devices
            ],
        },
        "runtime_profile": _runtime_profile(),
        "source_revision": _source_revision(),
        "source_tree_digest": _source_digest(),
        "measurement_scope": "synchronized host API including validation/transfers",
        "first_call_seconds": first_seconds,
        "compile_and_run_seconds": first_seconds,
        "first_iterations": first_iterations,
        "warm_seconds": timing["median_seconds"],
        **timing,
        "iterations": result.iterations,
        "error": result.error,
        "objective": value,
        "converged": result.converged,
        "memory": _memory_stats(devices),
        "memory_scope": "process lifetime; per-case only in isolated CLI worker",
        "reference_validation": "small family references; not a large-case certificate",
    }
    validate_record(record)
    return record


def benchmark_classical(n: int, dtype: str, device: str) -> dict[str, object]:
    x = np.linspace(-4.0, 4.0, n, dtype=dtype)
    cost = (x[:, None] - x[None, :]) ** 2
    a = np.exp(-0.5 * (x + 1.0) ** 2)
    b = np.exp(-0.5 * (x - 1.0) ** 2)
    a, b = a / a.sum(), b / b.sum()
    config = erot.SolverConfig(
        epsilon=0.5,
        tolerance=1e-8 if dtype == "float64" else 1e-5,
        max_iterations=20_000,
        dtype=dtype,  # type: ignore[arg-type]
        device=device,
    )

    def run() -> erot.SolveResult:
        return erot.solve(
            cost,
            [a, b],
            problem="classical",
            regularizer="shannon",
            method="sinkhorn",
            config=config,
        )

    return _measure(run, "classical-shannon", n, cost, [a, b], config)


def benchmark_quantum(n: int, dtype: str, device: str) -> dict[str, object]:
    real_dtype = np.float64 if dtype == "float64" else np.float32
    marginal = np.eye(n, dtype=real_dtype) / n
    generator = np.random.default_rng(1729 + n)
    raw_cost = generator.normal(size=(n * n, n * n)).astype(real_dtype)
    cost = (raw_cost + raw_cost.T) / (2 * np.sqrt(n * n))
    config = erot.SolverConfig(
        epsilon=1.0,
        tolerance=1e-8 if dtype == "float64" else 1e-5,
        max_iterations=20_000,
        dtype=dtype,  # type: ignore[arg-type]
        device=device,
    )

    def run() -> erot.SolveResult:
        return erot.solve(
            cost,
            [marginal, marginal],
            problem="quantum",
            regularizer="quadratic",
            method="cyclic",
            config=config,
        )

    return _measure(run, "quantum-quadratic", n, cost, [marginal, marginal], config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--classical-sizes", nargs="*", type=int, default=[100, 1000, 5000]
    )
    parser.add_argument("--quantum-sizes", nargs="*", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--max-regression", type=float, default=0.10)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    cases = [("classical", size) for size in args.classical_sizes]
    cases += [("quantum", size) for size in args.quantum_sizes]
    if not cases or any(size <= 0 for _, size in cases):
        parser.error("request at least one positive problem size")
    if len(set(cases)) != len(cases):
        parser.error("duplicate cases are not allowed")
    if args.worker and len(cases) != 1:
        parser.error("an isolated worker must run exactly one case")
    if args.max_regression < 0 or not np.isfinite(args.max_regression):
        parser.error("--max-regression must be finite and nonnegative")

    def run_all() -> list[dict[str, object]]:
        if not args.worker:
            records = []
            with tempfile.TemporaryDirectory(prefix="erot-benchmark-") as directory:
                for index, (family, size) in enumerate(cases):
                    output = Path(directory) / f"{index}.json"
                    other = "quantum" if family == "classical" else "classical"
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--device",
                        args.device,
                        "--dtype",
                        args.dtype,
                        f"--{family}-sizes",
                        str(size),
                        f"--{other}-sizes",
                        "--output",
                        str(output),
                    ]
                    if args.profile_dir:
                        command += [
                            "--profile-dir",
                            str(args.profile_dir.resolve() / f"{family}-{size}"),
                        ]
                    subprocess.run(command, check=True)
                    records.extend(json.loads(output.read_text())["records"])
            return records
        records = [
            benchmark_classical(size, args.dtype, args.device)
            for size in args.classical_sizes
        ]
        records.extend(
            benchmark_quantum(size, args.dtype, args.device)
            for size in args.quantum_sizes
        )
        for record in records:
            record["memory_scope"] = "fresh per-case process lifetime"
        return records

    if args.profile_dir and args.worker:
        args.profile_dir.mkdir(parents=True, exist_ok=True)
        with jax.profiler.trace(str(args.profile_dir)):
            records = run_all()
    else:
        records = run_all()
    report = {
        "schema_version": 2,
        "jax_version": jax.__version__,
        "dtype": args.dtype,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        baseline_records = {
            (record["problem"], record["size"]): record
            for record in baseline["records"]
        }
        regressions = []
        for record in records:
            key = (record["problem"], record["size"])
            if key not in baseline_records:
                raise SystemExit(f"baseline is missing requested case: {key}")
            previous = baseline_records[key]
            try:
                assert_comparable(record, previous)
            except ValueError as exc:
                raise SystemExit(f"invalid benchmark comparison: {exc}") from exc
            ratio = record["warm_seconds"] / previous["warm_seconds"] - 1
            if ratio > args.max_regression:
                regressions.append((key, ratio))
        if regressions:
            details = ", ".join(
                f"{problem}[{size}] +{ratio:.1%}"
                for (problem, size), ratio in regressions
            )
            raise SystemExit(f"performance regression exceeds limit: {details}")


if __name__ == "__main__":
    main()
