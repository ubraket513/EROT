from __future__ import annotations

import tomllib
from pathlib import Path

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
    project = Path(__file__).resolve().parents[2] / "pyproject.toml"
    expected_version = tomllib.loads(project.read_text())["project"]["version"]
    assert erot.__version__ == expected_version
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


@pytest.mark.parametrize("field", ["epsilon", "tolerance", "max_iterations"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_config_rejects_nonfinite_controls(field, value):
    with pytest.raises(ValueError):
        erot.SolverConfig(**{"epsilon": 1.0, field: value})


def test_config_rejects_fractional_iteration_limit():
    with pytest.raises(ValueError):
        erot.SolverConfig(epsilon=1.0, max_iterations=1.5)


def test_auto_device_preserves_explicit_array_placement(monkeypatch):
    import jax

    import erot.api

    device = jax.devices("cpu")[0]
    cost = jax.device_put(np.zeros((2, 2)), device)
    marginal = jax.device_put(np.array([0.5, 0.5]), device)

    def unexpected_resolution(spec):
        raise AssertionError("explicit input placement should select the device")

    monkeypatch.setattr(erot.api, "resolve_device", unexpected_resolution)
    result = erot.solve(
        cost,
        [marginal, marginal],
        problem="classical",
        regularizer="shannon",
        method="sinkhorn",
        config=config(device="auto"),
    )
    assert result.coupling.devices() == {device}
