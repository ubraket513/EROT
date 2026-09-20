from __future__ import annotations

import numpy as np
import pytest

import erot


def _config(epsilon: float = 0.5) -> erot.SolverConfig:
    return erot.SolverConfig(
        epsilon=epsilon,
        tolerance=1e-10,
        max_iterations=10_000,
        dtype="float64",
        device="cpu",
    )


def _assert_marginals(
    coupling: np.ndarray, marginals: list[np.ndarray], tolerance: float = 2e-9
) -> None:
    for axis, marginal in enumerate(marginals):
        axes = tuple(index for index in range(coupling.ndim) if index != axis)
        np.testing.assert_allclose(coupling.sum(axis=axes), marginal, atol=tolerance)


def test_shannon_sinkhorn_two_marginals() -> None:
    a = np.array([0.4, 0.6])
    b = np.array([0.5, 0.5])
    cost = np.array([[0.0, 1.0], [1.0, 0.0]])
    result = erot.solve(
        cost,
        [a, b],
        problem="classical",
        regularizer="shannon",
        method="sinkhorn",
        config=_config(),
    )
    coupling = np.asarray(result.coupling)
    assert result.converged
    assert np.all(coupling >= 0)
    _assert_marginals(coupling, [a, b])


def test_shannon_sinkhorn_three_marginals_with_zero_entries() -> None:
    marginals = [
        np.array([0.25, 0.75]),
        np.array([0.6, 0.4]),
        np.array([0.0, 1.0]),
    ]
    result = erot.solve(
        np.zeros((2, 2, 2)),
        marginals,
        problem="classical",
        regularizer="shannon",
        method="sinkhorn",
        config=_config(epsilon=1.0),
    )
    coupling = np.asarray(result.coupling)
    assert result.converged
    _assert_marginals(coupling, marginals)
    expected = np.einsum("i,j,k->ijk", *marginals)
    np.testing.assert_allclose(coupling, expected, atol=2e-9)


def test_quadratic_cyclic_projection() -> None:
    a = np.array([0.4, 0.6])
    b = np.array([0.5, 0.5])
    cost = np.array([[0.0, 1.0], [1.0, 0.0]])
    result = erot.solve(
        cost,
        [a, b],
        problem="classical",
        regularizer="quadratic",
        method="cyclic",
        config=_config(),
    )
    coupling = np.asarray(result.coupling)
    assert result.converged
    assert np.all(coupling >= 0)
    _assert_marginals(coupling, [a, b])
    np.testing.assert_allclose(coupling, [[0.4, 0.0], [0.1, 0.5]], atol=2e-9)


def test_float32_mode() -> None:
    result = erot.solve(
        np.zeros((2, 2)),
        [np.array([0.5, 0.5]), np.array([0.5, 0.5])],
        problem="classical",
        regularizer="shannon",
        method="sinkhorn",
        config=erot.SolverConfig(
            epsilon=1.0,
            tolerance=1e-5,
            max_iterations=100,
            dtype="float32",
            device="cpu",
        ),
    )
    assert result.coupling.dtype.name == "float32"
    assert result.converged


@pytest.mark.reference
def test_quadratic_matches_cvxpy_reference() -> None:
    cp = pytest.importorskip("cvxpy")
    a = np.array([0.2, 0.5, 0.3])
    b = np.array([0.4, 0.6])
    cost = np.array([[0.0, 0.7], [0.3, 0.1], [0.9, 0.2]])
    epsilon = 0.8
    result = erot.solve(
        cost,
        [a, b],
        problem="classical",
        regularizer="quadratic",
        method="cyclic",
        config=_config(epsilon=epsilon),
    )

    coupling = cp.Variable(cost.shape, nonneg=True)
    objective = cp.Minimize(
        cp.sum(cp.multiply(cost, coupling)) + epsilon * cp.sum_squares(coupling) / 2
    )
    reference = cp.Problem(
        objective,
        [cp.sum(coupling, axis=1) == a, cp.sum(coupling, axis=0) == b],
    )
    reference.solve(solver="CLARABEL")

    assert reference.status in {"optimal", "optimal_inaccurate"}
    np.testing.assert_allclose(result.coupling, coupling.value, atol=2e-6)
