"""Independent objective and marginal-derivative contracts for migration."""

from __future__ import annotations

import numpy as np
import pytest

import erot

pytestmark = pytest.mark.reference


def shannon_value(cost, plan, epsilon):
    positive = plan > 0
    entropy = np.sum(plan[positive] * (np.log(plan[positive]) - 1.0))
    return float(np.sum(cost * plan) + epsilon * entropy)


def reference_shannon(cost, a, b, epsilon):
    cp = pytest.importorskip("cvxpy")
    plan = cp.Variable(cost.shape, nonneg=True)
    row_constraint = cp.sum(plan, axis=1) == a
    col_constraint = cp.sum(plan, axis=0) == b
    objective = cp.Minimize(
        cp.sum(cp.multiply(cost, plan))
        - epsilon * cp.sum(cp.entr(plan))
        - epsilon * cp.sum(plan)
    )
    problem = cp.Problem(objective, [row_constraint, col_constraint])
    problem.solve(solver="CLARABEL")
    assert problem.status == "optimal"
    return float(problem.value), np.asarray(plan.value), -row_constraint.dual_value


def solve_shannon(cost, a, b, epsilon):
    return erot.solve(
        cost,
        [a, b],
        problem="classical",
        regularizer="shannon",
        method="sinkhorn",
        config=erot.SolverConfig(
            epsilon=epsilon,
            tolerance=1e-10,
            max_iterations=10000,
            dtype="float64",
            device="cpu",
        ),
    )


def test_shannon_objective_matches_independent_reference():
    cost = np.array([[0.0, 0.7], [0.3, 0.1], [0.9, 0.2]])
    a = np.array([0.2, 0.5, 0.3])
    b = np.array([0.4, 0.6])
    epsilon = 0.7
    result = solve_shannon(cost, a, b, epsilon)
    expected_value, expected_plan, _ = reference_shannon(cost, a, b, epsilon)
    assert result.converged
    np.testing.assert_allclose(result.coupling, expected_plan, atol=2e-5)
    assert shannon_value(cost, np.asarray(result.coupling), epsilon) == pytest.approx(
        expected_value, abs=2e-6
    )


def test_sinkhorn_potential_derivative_matches_reference():
    cost = np.array([[0.0, 0.7], [0.3, 0.1], [0.9, 0.2]])
    a = np.array([0.2, 0.5, 0.3])
    b = np.array([0.4, 0.6])
    epsilon = 0.7
    result = solve_shannon(cost, a, b, epsilon)
    assert result.converged
    stationarity = cost + epsilon * np.log(np.asarray(result.coupling))
    g = stationarity[0, :]
    f = stationarity[:, 0] - g[0]
    np.testing.assert_allclose(f[:, None] + g[None, :], stationarity, atol=2e-7)
    direction = np.array([1.0, -1.0, 0.0])
    h = 1e-3
    plus, _, _ = reference_shannon(cost, a + h * direction, b, epsilon)
    minus, _, _ = reference_shannon(cost, a - h * direction, b, epsilon)
    derivative = (plus - minus) / (2 * h)
    assert np.dot(f, direction) == pytest.approx(derivative, rel=2e-3, abs=1e-4)


def test_nonunit_mass_and_zero_support_preserved():
    result = solve_shannon(
        np.array([[0.0, 1.0], [2.0, 0.0]]),
        np.array([2.0, 0.0]),
        np.array([0.5, 1.5]),
        0.7,
    )
    assert result.converged
    np.testing.assert_allclose(result.coupling, [[0.5, 1.5], [0.0, 0.0]], atol=1e-8)
