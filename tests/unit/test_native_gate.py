"""Synthetic measurements test admission math; they are not performance claims."""

import math

import pytest

from benchmarks.native_gate import compare_candidate, estimate_speedup
from benchmarks.reporting import COMPARISON_FIELDS


def record(seconds=3.0):
    return {
        "schema_version": 2,
        "measurement_scope": "synchronized host API including validation/transfers",
        "problem": "classical-shannon",
        "case_digest": "example",
        "objective_convention": "coupling-entropy-minus-mass",
        "dtype": "float64",
        "geometry": "dense",
        "shape": [2, 3],
        "epsilon": 0.2,
        "tolerance": 1e-8,
        "max_iterations": 1000,
        "output_policy": "dense-coupling",
        "hardware_profile": {"devices": ["example"]},
        "runtime_profile": {"jax": "example"},
        "converged": True,
        "objective": -0.5,
        "error": 1e-10,
        "warm_seconds": seconds,
        "samples_seconds": [seconds] * 7,
    }


def peak(record, value, kind="isolated_peak_live_bytes"):
    return {
        "measurement_kind": kind,
        "scope": "full_solve",
        "peak_live_bytes": value,
        **{key: record[key] for key in COMPARISON_FIELDS},
    }


def test_amdahl_screen_and_overhead():
    assert estimate_speedup(3, 10, 1000)["upper_bound_without_overhead"] < 1.5
    assert estimate_speedup(8, 10, 4)["predicted_speedup"] == pytest.approx(2.5)
    assert estimate_speedup(8, 10, 4, 1)["predicted_speedup"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "args",
    [
        (11, 10, 2),
        (-1, 10, 2),
        (1, 0, 2),
        (1, 10, 0),
        (1, 10, math.inf),
        (1, 10, 2, -1),
        (math.nan, 10, 2),
    ],
)
def test_invalid_screen_measurements(args):
    with pytest.raises(ValueError):
        estimate_speedup(*args)


def test_full_solve_speed_gate():
    assert compare_candidate(record(2), record(3))["measurement_gate_pass"]
    assert not compare_candidate(record(2.1), record(3))["measurement_gate_pass"]


@pytest.mark.parametrize(
    "change",
    [
        {"converged": False},
        {"objective": 1},
        {"dtype": "float32"},
        {"samples_seconds": [3.0]},
    ],
)
def test_invalid_or_incomparable_candidate(change):
    with pytest.raises(ValueError):
        compare_candidate(record() | change, record())


def test_only_isolated_full_solve_peak_qualifies():
    base, candidate = record(), record()
    assert compare_candidate(
        candidate,
        base,
        baseline_peak=peak(base, 100),
        candidate_peak=peak(candidate, 75),
    )["measurement_gate_pass"]
    assert not compare_candidate(
        candidate,
        base,
        baseline_peak=peak(base, 100),
        candidate_peak=peak(candidate, 76),
    )["measurement_gate_pass"]
    for change in (
        {"measurement_kind": "compiler_bytes"},
        {"measurement_kind": "process_high_water"},
        {"scope": "kernel"},
        {"peak_live_bytes": 0},
        {"case_digest": "other"},
        {"epsilon": 0.5},
        {"tolerance": 1e-2},
        {"output_policy": "potentials-only"},
        {"flow": {"steps": 2}},
    ):
        with pytest.raises(ValueError):
            compare_candidate(
                candidate,
                base,
                baseline_peak=peak(base, 100),
                candidate_peak=peak(candidate, 75) | change,
            )


@pytest.mark.parametrize("scope", [None, "kernel-only", "device-core", "full solve"])
def test_reject_incomplete_or_unknown_timing_scope(scope):
    candidate, baseline = record(1), record(3)
    if scope is None:
        candidate.pop("measurement_scope")
    else:
        candidate["measurement_scope"] = scope
    with pytest.raises(ValueError, match="scope"):
        compare_candidate(candidate, baseline)
    with pytest.raises(ValueError, match="scope"):
        compare_candidate(baseline, candidate)
    with pytest.raises(ValueError, match="scope"):
        compare_candidate(candidate, candidate)
