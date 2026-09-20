"""Reject scientifically invalid or incomparable performance records."""

from __future__ import annotations

from copy import deepcopy

import pytest

from benchmarks.reporting import (
    assert_comparable,
    summarize_timings,
    validate_record,
)


def valid_record():
    return {
        "schema_version": 2,
        "problem": "classical-shannon",
        "case_digest": "example-input",
        "objective_convention": "coupling-entropy-minus-mass",
        "dtype": "float64",
        "geometry": "dense-squared-euclidean",
        "shape": [3, 2],
        "epsilon": 0.7,
        "tolerance": 1e-8,
        "max_iterations": 10000,
        "output_policy": "dense-coupling",
        "hardware_profile": {"devices": ["cpu"], "allocated_cpus": 8},
        "runtime_profile": {"jax": "test-runtime"},
        "source_revision": "old",
        "converged": True,
        "objective": -0.5,
        "error": 1e-10,
        "warm_seconds": 4.0,
        "samples_seconds": [1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0],
    }


def test_summary_uses_median_and_preserves_samples():
    samples = [1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0]
    result = summarize_timings(samples)
    assert result == {
        "median_seconds": 4.0,
        "min_seconds": 1.0,
        "max_seconds": 9.0,
        "samples_seconds": samples,
    }


@pytest.mark.parametrize("samples", [[], [float("nan")], [float("inf")], [-1.0], [0.0]])
def test_invalid_samples_are_rejected(samples):
    with pytest.raises(ValueError):
        summarize_timings(samples)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("converged", False),
        ("objective", float("nan")),
        ("error", float("inf")),
        ("error", -1.0),
        ("warm_seconds", 0.0),
        ("samples_seconds", [float("nan")]),
        ("samples_seconds", []),
        ("schema_version", 1),
    ],
)
def test_invalid_outcomes_cannot_win_comparisons(field, value):
    candidate = valid_record()
    candidate[field] = value
    with pytest.raises(ValueError):
        validate_record(candidate)
    with pytest.raises(ValueError):
        assert_comparable(candidate, valid_record())


@pytest.mark.parametrize(
    "field",
    [
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
    ],
)
def test_comparison_rejects_changed_or_missing_contract(field):
    baseline = valid_record()
    changed = deepcopy(baseline)
    changed[field] = "different"
    with pytest.raises(ValueError):
        assert_comparable(changed, baseline)
    missing = deepcopy(baseline)
    del missing[field]
    with pytest.raises(ValueError):
        assert_comparable(missing, baseline)


def test_source_revision_is_recorded_but_may_differ():
    baseline = valid_record()
    candidate = deepcopy(baseline)
    candidate["source_revision"] = "new"
    validate_record(candidate)
    assert_comparable(candidate, baseline)


def test_flow_contracts_are_not_implicitly_comparable():
    baseline = valid_record()
    baseline["flow"] = {
        "energy": "entropy",
        "time_step": 0.1,
        "physical_horizon": 1.0,
        "inner_tolerance": 1e-8,
        "checkpoint_policy": "every-10",
    }
    changed = deepcopy(baseline)
    changed["flow"]["time_step"] = 0.2
    with pytest.raises(ValueError):
        assert_comparable(changed, baseline)
    del changed["flow"]
    with pytest.raises(ValueError):
        assert_comparable(changed, baseline)
