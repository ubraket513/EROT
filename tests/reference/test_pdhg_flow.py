"""Unregularized JKO is checked against its own convex coupling objective."""

import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from erot.flows.functionals import Entropy, Quadratic
from erot.flows.pdhg import coupled_operator_norm_squared, solve_pdhg_jko
from erot.solvers.state import CONVERGED, INVALID_INPUT, ITERATION_LIMIT

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("kind", ["quadratic", "entropy"])
def test_pdhg_matches_independent_jko_reference(kind):
    x = np.linspace(0, 1, 4)
    cost = (x[:, None] - x[None, :]) ** 2
    previous = np.array([0.1, 0.5, 0.3, 0.1])
    target = np.array([0.4, 0.2, 0.1, 0.3])
    dt = 0.4
    energy = (
        Quadratic(jnp.asarray(target), 3.0)
        if kind == "quadratic"
        else Entropy(jnp.full(4, 0.25))
    )
    plan = cp.Variable((4, 4), nonneg=True)
    rho = cp.sum(plan, axis=1)
    value = (
        1.5 * cp.sum_squares(rho - target)
        if kind == "quadratic"
        else -cp.sum(cp.entr(rho)) - np.log(0.25) * cp.sum(rho) - cp.sum(rho)
    )
    reference = cp.Problem(
        cp.Minimize(dt * value + 0.5 * cp.sum(cp.multiply(cost, plan))),
        [cp.sum(plan, axis=0) == previous],
    )
    reference.solve(solver="CLARABEL")
    assert reference.status == "optimal"
    state, d = jax.jit(solve_pdhg_jko)(
        jnp.asarray(cost), jnp.asarray(previous), energy, dt, 1e-8, 30000
    )
    assert d.status == CONVERGED
    assert d.feasibility <= 1e-8 and d.stationarity <= 1e-8
    assert np.min(state.coupling) >= 0 and np.min(state.rho) >= 0
    np.testing.assert_allclose(state.rho, rho.value, atol=2e-5)
    assert float(d.objective) == pytest.approx(reference.value, abs=2e-7)


def test_operator_norm_is_for_coupling_and_density():
    n, m = 3, 2
    operator = np.zeros((n + m, n * m + n))
    for i in range(n):
        for j in range(m):
            operator[i, i * m + j] = 1
            operator[n + j, i * m + j] = 1
        operator[i, n * m + i] = -1
    exact = np.linalg.norm(operator, 2) ** 2
    assert float(coupled_operator_norm_squared(n, m)) == pytest.approx(exact)


def test_bad_steps_and_insufficient_work_are_explicit():
    c = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    previous = jnp.array([0.9, 0.1])
    energy = Quadratic(jnp.array([0.1, 0.9]), 10.0)
    _, d = solve_pdhg_jko(
        c, previous, energy, 0.5, 1e-9, 1, primal_step=10.0, dual_step=10.0
    )
    assert d.status == INVALID_INPUT
    _, d = solve_pdhg_jko(c, previous, energy, 0.5, 1e-9, 1)
    assert d.status == ITERATION_LIMIT


def test_rectangular_pdhg_resume_and_extrapolation():
    c = jnp.array([[0.0, 0.4, 0.9], [0.7, 0.2, 0.0]])
    p = jnp.array([0.2, 0.5, 0.3])
    e = Quadratic(jnp.array([0.6, 0.4]))
    first, _ = solve_pdhg_jko(c, p, e, 0.2, 1e-12, 3, theta=0.5)
    continued, _ = solve_pdhg_jko(c, p, e, 0.2, 1e-12, 7, theta=0.5, state=first)
    whole, _ = solve_pdhg_jko(c, p, e, 0.2, 1e-12, 10, theta=0.5)
    for a, b in zip(jax.tree.leaves(continued), jax.tree.leaves(whole), strict=True):
        np.testing.assert_array_equal(a, b)
