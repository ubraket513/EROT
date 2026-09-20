"""Scientific comparison gates shared by benchmark drivers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import median

COMPARISON_FIELDS = (
    "schema_version",
    "problem",
    "case_digest",
    "objective_convention",
    "dtype",
    "geometry",
    "shape",
    "epsilon",
    "tolerance",
    "max_iterations",
    "output_policy",
    "hardware_profile",
    "runtime_profile",
)


def summarize_timings(samples_seconds: Sequence[float]) -> dict[str, object]:
    """Summarize already synchronized, finite, positive elapsed times."""
    samples = [float(value) for value in samples_seconds]
    if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("timing samples must be nonempty, finite, and positive")
    return {
        "median_seconds": median(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "samples_seconds": samples,
    }


def validate_record(record: Mapping[str, object]) -> None:
    """Reject failed numerical outcomes and invalid performance measurements."""
    if record.get("schema_version") != 2:
        raise ValueError("benchmark record must use schema version 2")
    if record.get("converged") is not True:
        raise ValueError("nonconverged solves cannot enter performance comparisons")
    try:
        objective = float(record["objective"])
        residual = float(record["error"])
        seconds = float(record["warm_seconds"])
        tolerance = float(record["tolerance"])
        summary = summarize_timings(record["samples_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing or invalid benchmark measurement: {exc}") from exc
    if not all(
        math.isfinite(value) for value in (objective, residual, seconds, tolerance)
    ):
        raise ValueError("benchmark measurements must be finite")
    if residual < 0 or tolerance <= 0 or residual > tolerance:
        raise ValueError("residual does not meet the declared tolerance")
    if seconds <= 0 or not math.isclose(
        seconds, summary["median_seconds"], rel_tol=1e-12
    ):
        raise ValueError("warm_seconds must be the positive median of recorded samples")


def assert_comparable(
    candidate: Mapping[str, object], baseline: Mapping[str, object]
) -> None:
    """Require the same scientific and environment contract, allowing new code."""
    validate_record(candidate)
    validate_record(baseline)
    for field in COMPARISON_FIELDS:
        if field not in candidate or field not in baseline:
            raise ValueError(f"missing comparison field: {field}")
        if candidate[field] != baseline[field]:
            raise ValueError(f"incompatible comparison field: {field}")
    if candidate.get("flow") != baseline.get("flow"):
        raise ValueError("incompatible flow contract")
