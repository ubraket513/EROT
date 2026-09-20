"""Conditional Amdahl screening and matched full-solve native measurement gates."""

import argparse
import hashlib
import json
import math
from pathlib import Path

if __package__:
    from .reporting import COMPARISON_FIELDS, assert_comparable
else:
    from reporting import COMPARISON_FIELDS, assert_comparable


def _number(value, name, minimum=0, strict=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value) or value < minimum or (strict and value == minimum):
        raise ValueError(f"{name} is outside its valid range")
    return value


def estimate_speedup(
    operation_seconds, solve_seconds, kernel_speedup, overhead_seconds=0
):
    """A conditional estimate from non-overlapping critical-path profile times.

    This is not a measured native speedup. Kernel speedup and integration
    overhead are supplied assumptions; a complete candidate still needs timing.
    """
    operation = _number(operation_seconds, "operation_seconds")
    total = _number(solve_seconds, "solve_seconds", strict=True)
    factor = _number(kernel_speedup, "kernel_speedup", strict=True)
    overhead = _number(overhead_seconds, "overhead_seconds")
    if operation > total:
        raise ValueError(
            "operation duration exceeds solve duration; check scope/overlap"
        )
    fraction = operation / total
    prediction = total / (total - operation + operation / factor + overhead)
    return {
        "interpretation": "conditional prediction, not a measured result",
        "operation_fraction": fraction,
        "screen_fraction_pass": fraction >= 0.3,
        "predicted_speedup": prediction,
        "predicted_speed_gate_pass": prediction >= 1.5,
        "upper_bound_without_overhead": None if fraction == 1 else 1 / (1 - fraction),
        "upper_bound_unbounded": fraction == 1,
        "assumed_kernel_speedup": factor,
        "assumed_overhead_seconds": overhead,
    }


def _peak(value, record):
    if (
        not isinstance(value, dict)
        or value.get("measurement_kind") != "isolated_peak_live_bytes"
        or value.get("scope") != "full_solve"
    ):
        raise ValueError(
            "memory evidence must measure isolated full-solve peak live bytes"
        )
    for key in COMPARISON_FIELDS:
        if key not in value or value[key] != record[key]:
            raise ValueError(f"incompatible peak-memory identity: {key}")
    if value.get("flow") != record.get("flow"):
        raise ValueError("incompatible peak-memory flow contract")
    return _number(value.get("peak_live_bytes"), "peak_live_bytes", strict=True)


def compare_candidate(
    candidate,
    baseline,
    objective_atol=1e-8,
    objective_rtol=1e-6,
    baseline_peak=None,
    candidate_peak=None,
):
    """Check measurement admission only; ABI/platform/transform review is separate."""
    assert_comparable(candidate, baseline)
    scope = "synchronized host API including validation/transfers"
    if any(
        record.get("measurement_scope") != scope for record in (candidate, baseline)
    ):
        raise ValueError(
            "native admission requires full-solve measurement_scope: " + scope
        )
    atol = _number(objective_atol, "objective_atol")
    rtol = _number(objective_rtol, "objective_rtol")
    if min(len(candidate["samples_seconds"]), len(baseline["samples_seconds"])) < 7:
        raise ValueError(
            "native admission requires at least seven synchronized samples"
        )
    if not math.isclose(
        candidate["objective"], baseline["objective"], rel_tol=rtol, abs_tol=atol
    ):
        raise ValueError("candidate objective disagrees with the matched baseline")
    speedup = baseline["warm_seconds"] / candidate["warm_seconds"]
    reduction = None
    if (baseline_peak is None) != (candidate_peak is None):
        raise ValueError("both baseline and candidate peak evidence are required")
    if baseline_peak is not None:
        reduction = 1 - _peak(candidate_peak, candidate) / _peak(
            baseline_peak, baseline
        )
    speed_pass = speedup >= 1.5
    memory_pass = reduction is not None and reduction >= 0.25
    return {
        "scope": "measurement gate only; not backend release certification",
        "measurement_gate_pass": speed_pass or memory_pass,
        "full_solve_speedup": speedup,
        "speed_gate_pass": speed_pass,
        "peak_live_memory_reduction": reduction,
        "memory_gate_pass": memory_pass,
        "objective_atol": atol,
        "objective_rtol": rtol,
        "remaining_checks": [
            "representative workload regressions",
            "required capability justification if used instead of performance",
            "architecture/runtime/ABI and optional installation",
            "stream/layout/alias/error contracts",
            "batching/sharding/derivative support or explicit rejection",
            "verified fallback and precision",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    screen = commands.add_parser("screen", help="conditional Amdahl prediction")
    screen.add_argument("--operation-seconds", required=True, type=float)
    screen.add_argument("--solve-seconds", required=True, type=float)
    screen.add_argument("--kernel-speedup", required=True, type=float)
    screen.add_argument("--overhead-seconds", type=float, default=0)
    compare = commands.add_parser("compare", help="matched schema2 full-solve reports")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--baseline-peak", type=Path)
    compare.add_argument("--candidate-peak", type=Path)
    compare.add_argument("--objective-atol", type=float, default=1e-8)
    compare.add_argument("--objective-rtol", type=float, default=1e-6)
    args = parser.parse_args(argv)
    try:
        if args.command == "screen":
            result = estimate_speedup(
                args.operation_seconds,
                args.solve_seconds,
                args.kernel_speedup,
                args.overhead_seconds,
            )
        else:
            evidence = {}

            def read(path):
                if path is None:
                    return None
                content = path.read_bytes()
                evidence[str(path)] = hashlib.sha256(content).hexdigest()
                return json.loads(content)

            baseline, candidate = read(args.baseline), read(args.candidate)
            result = compare_candidate(
                candidate,
                baseline,
                args.objective_atol,
                args.objective_rtol,
                read(args.baseline_peak),
                read(args.candidate_peak),
            )
            result["evidence_sha256"] = evidence
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 1 if args.command == "compare" and not result["measurement_gate_pass"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
