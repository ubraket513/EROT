"""Finite-epsilon JKO must match the joint entropic coupling problem."""

import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from erot.flows.functionals import Entropy, Quadratic
from erot.flows.jko import solve_entropic_jko
from erot.flows.state import INNER_SOLVE_FAILED
from erot.solvers.state import CONVERGED, ITERATION_LIMIT

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("kind", ["quadratic", "entropy"])
def test_entropic_jko_matches_independent_objective(kind):
    x = np.array([0.0, 0.5, 1.0])
    c = (x[:, None] - x[None, :]) ** 2
    p = np.array([0.2, 0.5, 0.3])
    dt = 0.3
    eps = 0.15
    target = np.array([0.5, 0.2, 0.3])
    energy = (
        Quadratic(jnp.asarray(target), 2.0)
        if kind == "quadratic"
        else Entropy(jnp.full(3, 1 / 3))
    )
    pi = cp.Variable((3, 3), nonneg=True)
    rho = cp.sum(pi, axis=1)
    f = (
        cp.sum_squares(rho - target)
        if kind == "quadratic"
        else -cp.sum(cp.entr(rho)) + np.log(3) * cp.sum(rho) - cp.sum(rho)
    )
    objective = f + (
        cp.sum(cp.multiply(c, pi)) - eps * cp.sum(cp.entr(pi)) - eps * cp.sum(pi)
    ) / (2 * dt)
    reference = cp.Problem(cp.Minimize(objective), [cp.sum(pi, axis=0) == p])
    reference.solve(solver="CLARABEL")
    assert reference.status == "optimal"
    state, d = jax.jit(solve_entropic_jko)(
        jnp.asarray(c), jnp.asarray(p), energy, dt, eps, 1e-7, 500
    )
    assert d.status == CONVERGED
    assert d.stationarity <= 1e-7 and d.feasibility <= 1e-7
    np.testing.assert_allclose(state.rho, rho.value, atol=2e-5)
    assert float(d.objective) == pytest.approx(reference.value, abs=2e-7)
    assert state.inner_iterations > 0


def test_unconverged_transport_cannot_pass_outer_solver():
    c = jnp.array([[0.0, 3.0], [3.0, 0.0]])
    p = jnp.array([0.99, 0.01])
    energy = Quadratic(jnp.array([0.1, 0.9]))
    state, d = solve_entropic_jko(c, p, energy, 0.1, 0.01, 1e-8, 20, inner_iterations=1)
    assert d.status == INNER_SOLVE_FAILED
    assert state.iterations == 0


def test_plain_projected_gradient_option_and_work_limit():
    c = jnp.array([[0.0, 0.3], [0.3, 0.0]])
    p = jnp.array([0.3, 0.7])
    energy = Quadratic(jnp.array([0.6, 0.4]))
    _, d = solve_entropic_jko(c, p, energy, 0.3, 0.2, 1e-7, 500, method="sgd")
    assert d.status == CONVERGED
    _, d = solve_entropic_jko(c, p, energy, 0.3, 0.2, 1e-12, 0, inner_tolerance=1e-14)
    assert d.status == ITERATION_LIMIT


def test_entropic_resume_matches_fixed_uninterrupted_work():
    c = jnp.array([[0.0, 0.3], [0.3, 0.0]])
    p = jnp.array([0.3, 0.7])
    e = Quadratic(jnp.array([0.6, 0.4]))
    first, _ = solve_entropic_jko(c, p, e, 0.3, 0.2, 1e-10, 2, inner_tolerance=1e-12)
    resumed, _ = solve_entropic_jko(
        c, p, e, 0.3, 0.2, 1e-10, 3, inner_tolerance=1e-12, state=first
    )
    whole, _ = solve_entropic_jko(c, p, e, 0.3, 0.2, 1e-10, 5, inner_tolerance=1e-12)
    np.testing.assert_allclose(resumed.rho, whole.rho, atol=1e-12)
    assert resumed.iterations == whole.iterations
    assert resumed.inner_iterations == whole.inner_iterations
