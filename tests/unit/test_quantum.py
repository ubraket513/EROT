from __future__ import annotations

import numpy as np
import pytest

import erot
from erot.quantum import partial_trace_first, partial_trace_second


def _config() -> erot.SolverConfig:
    return erot.SolverConfig(
        epsilon=0.75,
        tolerance=1e-9,
        max_iterations=20_000,
        dtype="float64",
        device="cpu",
    )


def test_partial_traces_of_product_state() -> None:
    rho_a = np.array([[0.7, 0.1], [0.1, 0.3]])
    rho_b = np.array([[0.6, 0.0], [0.0, 0.4]])
    product = np.kron(rho_a, rho_b)
    np.testing.assert_allclose(partial_trace_second(product, 2, 2), rho_a)
    np.testing.assert_allclose(partial_trace_first(product, 2, 2), rho_b)


def test_quantum_zero_cost_uniform_solution() -> None:
    rho_a = np.eye(2) / 2
    rho_b = np.eye(2) / 2
    result = erot.solve(
        np.zeros((4, 4)),
        [rho_a, rho_b],
        problem="quantum",
        regularizer="quadratic",
        method="cyclic",
        config=_config(),
    )
    coupling = np.asarray(result.coupling)
    assert result.converged
    np.testing.assert_allclose(coupling, np.eye(4) / 4, atol=1e-9)
    np.testing.assert_allclose(coupling, coupling.conj().T, atol=1e-12)
    assert np.linalg.eigvalsh(coupling).min() >= -1e-12


def test_quantum_supports_different_subsystem_dimensions() -> None:
    rho_a = np.eye(2) / 2
    rho_b = np.eye(3) / 3
    result = erot.solve(
        np.zeros((6, 6)),
        [rho_a, rho_b],
        problem="quantum",
        regularizer="quadratic",
        method="cyclic",
        config=_config(),
    )
    assert result.converged
    np.testing.assert_allclose(result.coupling, np.eye(6) / 6, atol=1e-9)


def test_quantum_complex_pure_marginals() -> None:
    plus = np.array([1.0, 1.0j]) / np.sqrt(2)
    zero = np.array([1.0, 0.0])
    rho_a = np.outer(plus, plus.conj())
    rho_b = np.outer(zero, zero.conj())
    expected = np.kron(rho_a, rho_b)
    result = erot.solve(
        np.zeros((4, 4), dtype=np.complex128),
        [rho_a, rho_b],
        problem="quantum",
        regularizer="quadratic",
        method="cyclic",
        config=_config(),
    )
    assert result.converged
    np.testing.assert_allclose(result.coupling, expected, atol=2e-8)


@pytest.mark.reference
def test_quantum_matches_cvxpy_reference() -> None:
    cp = pytest.importorskip("cvxpy")
    rho_a = np.array([[0.65, 0.1], [0.1, 0.35]])
    rho_b = np.array([[0.55, 0.0], [0.0, 0.45]])
    cost = np.diag([0.0, 0.3, 0.8, 0.1])
    epsilon = 0.75

    result = erot.solve(
        cost,
        [rho_a, rho_b],
        problem="quantum",
        regularizer="quadratic",
        method="cyclic",
        config=_config(),
    )

    gamma = cp.Variable((4, 4), hermitian=True)
    constraints = [gamma >> 0]
    for i in range(2):
        for j in range(2):
            constraints.append(
                sum(gamma[i * 2 + a, j * 2 + a] for a in range(2)) == rho_a[i, j]
            )
    for a in range(2):
        for b in range(2):
            constraints.append(
                sum(gamma[i * 2 + a, i * 2 + b] for i in range(2)) == rho_b[a, b]
            )
    objective = cp.Minimize(
        cp.real(cp.trace(cost @ gamma)) + epsilon * cp.sum_squares(cp.abs(gamma)) / 2
    )
    problem = cp.Problem(objective, constraints)
    problem.solve(solver="CLARABEL")

    assert problem.status in {"optimal", "optimal_inaccurate"}
    coupling = np.asarray(result.coupling)
    actual_objective = (
        np.real(np.trace(cost @ coupling)) + epsilon * np.sum(np.abs(coupling) ** 2) / 2
    )
    assert actual_objective == pytest.approx(problem.value, rel=2e-5, abs=2e-6)
    np.testing.assert_allclose(coupling, gamma.value, atol=2e-5)


def test_quantum_rejects_non_psd_marginal() -> None:
    with pytest.raises(ValueError, match="positive semidefinite"):
        erot.solve(
            np.zeros((4, 4)),
            [np.diag([1.2, -0.2]), np.eye(2) / 2],
            problem="quantum",
            regularizer="quadratic",
            method="cyclic",
            config=_config(),
        )


def test_host_accepted_near_unit_trace_reaches_quantum_core():
    marginal = np.eye(2) * (1.0 + 1.5e-9) / 2
    result = erot.solve(
        np.zeros((4, 4)),
        [marginal, marginal],
        problem="quantum",
        regularizer="quadratic",
        method="cyclic",
        config=erot.SolverConfig(0.7, tolerance=1e-8, device="cpu"),
    )
    assert result.converged
    assert result.iterations > 0
