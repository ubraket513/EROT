"""State, domain and complex derivative contracts for entropy QOT."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from erot.solvers.quantum_entropy import entropy_dual, solve_quantum_entropy
from erot.solvers.state import CONVERGED, INVALID_INPUT, ITERATION_LIMIT

jax.config.update("jax_enable_x64", True)


def case():
    a = jnp.array([[0.6, 0.1j], [-0.1j, 0.4]])
    b = jnp.diag(jnp.array([0.2, 0.3, 0.5]))
    return jnp.zeros((6, 6)), a, b


def test_zero_cost_product_and_objective_constant():
    c, a, b = case()
    s, d = jax.jit(solve_quantum_entropy)(c, a, b, 0.4, 1e-9, 3000)
    assert d.status == CONVERGED
    np.testing.assert_allclose(s.coupling, np.kron(a, b), atol=2e-9)
    eigenvalues = np.linalg.eigvalsh(np.kron(a, b))
    expected = 0.4 * (np.sum(eigenvalues * np.log(eigenvalues)) - 1)
    np.testing.assert_allclose(d.primal, expected, atol=2e-9)
    np.testing.assert_allclose(d.dual, expected, atol=2e-9)
    assert d.feasibility <= 1e-9 and abs(d.gap) <= 1e-9


def test_exact_resume_and_zero_one_iteration_caps():
    c, a, b = case()
    s0, d0 = solve_quantum_entropy(c, a, b, 0.4, 1e-14, 0)
    assert s0.iterations == 0 and d0.status == ITERATION_LIMIT
    s1, d1 = solve_quantum_entropy(c, a, b, 0.4, 1e-14, 1)
    assert s1.iterations == 1 and d1.status == ITERATION_LIMIT
    resumed, _ = solve_quantum_entropy(c, a, b, 0.4, 1e-14, 6, state=s1)
    whole, _ = solve_quantum_entropy(c, a, b, 0.4, 1e-14, 7)
    for left, right in zip(jax.tree.leaves(resumed), jax.tree.leaves(whole)):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("change", ["rank", "trace", "epsilon", "nan", "budget"])
def test_invalid_domain_is_explicit(change):
    c, a, b = case()
    eps, budget = 0.4, 10
    if change == "rank":
        a = jnp.diag(jnp.array([1.0, 0.0]))
    elif change == "trace":
        a = a * 2
    elif change == "epsilon":
        eps = 0.0
    elif change == "nan":
        c = c.at[0, 0].set(jnp.nan)
    else:
        budget = 2**31
    _, d = solve_quantum_entropy(c, a, b, eps, 1e-8, budget)
    assert d.status == INVALID_INPUT


def test_complex_directional_dual_derivative():
    c, a, b = case()
    u = jnp.array([[0.2, 0.03 + 0.04j], [0.03 - 0.04j, -0.2]])
    v = jnp.diag(jnp.array([0.1, -0.2, 0.1]))
    h = jnp.array([[0.4, -0.1j], [0.1j, -0.4]])
    k = jnp.diag(jnp.array([0.1, 0.3, -0.4]))
    value, coupling, gradient = entropy_dual(c, a, b, 0.4, (u, v))
    step = 1e-6
    plus = entropy_dual(c, a, b, 0.4, (u + step * h, v + step * k))[0]
    minus = entropy_dual(c, a, b, 0.4, (u - step * h, v - step * k))[0]
    expected = jnp.real(jnp.vdot(gradient[0], h) + jnp.vdot(gradient[1], k))
    np.testing.assert_allclose((plus - minus) / (2 * step), expected, atol=1e-9)
    assert jnp.isfinite(value) and coupling.shape == (6, 6)


def test_batched_solver_and_iteration_overflow_rejection():
    c, a, b = case()
    _, d = jax.jit(
        jax.vmap(lambda cost: solve_quantum_entropy(cost, a, b, 0.4, 1e-7, 2000))
    )(jnp.stack([c, c + 0.2 * jnp.eye(6)]))
    np.testing.assert_array_equal(d.status, [CONVERGED, CONVERGED])
    initial, _ = solve_quantum_entropy(c, a, b, 0.4, 1e-8, 0)
    invalid = initial._replace(iterations=jnp.asarray(2**31 - 1, jnp.int64))
    _, diag = solve_quantum_entropy(c, a, b, 0.4, 1e-8, 1, state=invalid)
    assert diag.status == INVALID_INPUT


def test_float32_rounded_budget_and_resume_counter_are_invalid():
    c, a, b = case()
    _, diag = solve_quantum_entropy(c, a, b, 0.4, 1e-8, jnp.float32(2**31))
    assert diag.status == INVALID_INPUT
    state, _ = solve_quantum_entropy(c, a, b, 0.4, 1e-8, 0)
    state = state._replace(iterations=jnp.float32(2**31))
    _, diag = solve_quantum_entropy(c, a, b, 0.4, 1e-8, 0, state=state)
    assert diag.status == INVALID_INPUT


def test_large_identity_cost_shift_preserves_coupling_and_convergence():
    c, a, b = case()
    base, base_diag = solve_quantum_entropy(c, a, b, 0.4, 1e-8, 1000)
    shifted, diag = solve_quantum_entropy(c + 1e12 * jnp.eye(6), a, b, 0.4, 1e-8, 1000)
    assert diag.status == CONVERGED
    np.testing.assert_allclose(shifted.coupling, base.coupling, atol=1e-10)
    assert shifted.iterations == base.iterations
    np.testing.assert_allclose(diag.dual - base_diag.dual, 1e12, rtol=1e-15)
    np.testing.assert_allclose(diag.gap, base_diag.gap, atol=1e-10)


@pytest.mark.parametrize("count,budget", [(2**31, 0.0), (2**31 - 1, 1.0)])
def test_float_budget_cannot_wrap_wide_resume_counter(count, budget):
    c, a, b = case()
    state, _ = solve_quantum_entropy(c, a, b, 0.4, 1e-8, 0)
    state = state._replace(iterations=jnp.int64(count))
    _, diag = solve_quantum_entropy(
        c, a, b, 0.4, 1e-8, jnp.float32(budget), state=state
    )
    assert diag.status == INVALID_INPUT


def test_dual_evaluation_large_cost_shift_preserves_gradient():
    c, a, b = case()
    duals = (
        jnp.diag(jnp.array([0.123, -0.123])),
        jnp.diag(jnp.array([0.1, 0.2, -0.3])),
    )
    base = entropy_dual(c, a, b, 0.4, duals)
    shifted = entropy_dual(c + 1e12 * jnp.eye(6), a, b, 0.4, duals)
    np.testing.assert_allclose(shifted[1], base[1], atol=1e-12)
    for actual, expected in zip(shifted[2], base[2]):
        np.testing.assert_allclose(actual, expected, atol=1e-12)
