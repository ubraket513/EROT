"""Projection benchmark checks an independent numerical reference."""

import pytest

from benchmarks.benchmark_quantum_projection import benchmark_projection, memory_budget


def test_budget_does_not_mislabel_one_matrix_as_solver_peak():
    budget = memory_budget(128, 128, "complex128")
    assert budget["matrix_bytes"] == 4 * 1024**3
    assert budget["persistent_solver_arrays_bytes"] == 5 * budget["matrix_bytes"]
    assert budget["eigensolver_workspace_bytes"] is None
    assert budget["measured_peak_bytes"] is None


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_cpu_projection_matches_numpy_reference(dtype):
    report = benchmark_projection(4, dtype, "cpu", 17)
    assert report["validated"]
    assert report["dtype"] == dtype
    assert report["reference_relative_error"] < report["tolerance"]
    assert report["minimum_eigenvalue"] >= -report["tolerance"]
    assert len(report["projection"]["samples_seconds"]) == 7
    assert len(report["eigh"]["samples_seconds"]) == 7
    assert report["devices"][0]["platform"] == "cpu"


@pytest.mark.parametrize("dimensions", [(0, 2), (-1, 3), (2, 0)])
def test_budget_rejects_invalid_dimensions(dimensions):
    with pytest.raises(ValueError):
        memory_budget(*dimensions, "complex128")
