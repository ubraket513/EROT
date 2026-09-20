from __future__ import annotations

import numpy as np
import pytest

import erot


def config(**overrides: object) -> erot.SolverConfig:
    values: dict[str, object] = {
        "epsilon": 0.5,
        "tolerance": 1e-9,
        "max_iterations": 5_000,
        "dtype": "float64",
        "device": "cpu",
    }
    values.update(overrides)
    return erot.SolverConfig(**values)  # type: ignore[arg-type]


def test_public_exports() -> None:
    assert erot.__version__ == "0.1.0"
    assert set(erot.__all__) == {"SolveResult", "SolverConfig", "solve"}


@pytest.mark.parametrize(
    ("field", "value"),
    [("epsilon", 0), ("tolerance", 0), ("max_iterations", 0)],
)
def test_config_rejects_nonpositive_values(field: str, value: int) -> None:
    values = {"epsilon": 1.0, field: value}
    with pytest.raises(ValueError):
        erot.SolverConfig(**values)


def test_rejects_mismatched_classical_masses() -> None:
    with pytest.raises(ValueError, match="same total mass"):
        erot.solve(
            np.zeros((2, 2)),
            [np.array([0.5, 0.5]), np.array([0.2, 0.2])],
            problem="classical",
            regularizer="shannon",
            method="sinkhorn",
            config=config(),
        )


def test_reports_nonconvergence() -> None:
    result = erot.solve(
        np.array([[0.0, 5.0], [5.0, 0.0]]),
        [np.array([0.9, 0.1]), np.array([0.1, 0.9])],
        problem="classical",
        regularizer="shannon",
        method="sinkhorn",
        config=config(max_iterations=1, tolerance=1e-14),
    )
    assert result.iterations == 1
    assert not result.converged


def test_rejects_unsupported_solver_combination() -> None:
    with pytest.raises(ValueError, match="supported classical combinations"):
        erot.solve(
            np.zeros((2, 2)),
            [np.array([0.5, 0.5]), np.array([0.5, 0.5])],
            problem="classical",
            regularizer="quadratic",
            method="sinkhorn",
            config=config(),
        )
