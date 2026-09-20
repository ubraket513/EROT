"""Dykstra restart requires all corrections, not only a coupling snapshot."""

import jax
import jax.numpy as jnp
import numpy as np

from erot.solvers.quantum import solve_quantum_quadratic
from erot.solvers.state import CONVERGED, INVALID_INPUT

jax.config.update("jax_enable_x64", True)


def case():
    rng = np.random.default_rng(11)
    z = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    cost = jnp.asarray((z + z.conj().T) / 2)
    a = jnp.array([[0.5, 0.1j], [-0.1j, 0.5]])
    b = jnp.diag(jnp.array([0.2, 0.3, 0.5])).astype(complex)
    return cost, a, b


def test_quantum_resume_preserves_all_dykstra_corrections():
    cost, a, b = case()
    first, _ = solve_quantum_quadratic(cost, a, b, 0.7, 1e-12, 3)
    resumed, rd = solve_quantum_quadratic(cost, a, b, 0.7, 1e-12, 5, state=first)
    whole, wd = solve_quantum_quadratic(cost, a, b, 0.7, 1e-12, 8)
    for left, right in zip(
        jax.tree.leaves(resumed), jax.tree.leaves(whole), strict=True
    ):
        np.testing.assert_array_equal(left, right)
    assert rd.error == wd.error
    assert resumed.iterations == 8


def test_quantum_core_preserves_dense_reference_and_transforms():
    import erot

    cost, a, b = case()
    state, d = jax.jit(solve_quantum_quadratic)(cost, a, b, 0.7, 1e-9, 10000)
    ref = erot.solve(
        cost,
        [a, b],
        problem="quantum",
        regularizer="quadratic",
        method="cyclic",
        config=erot.SolverConfig(0.7, tolerance=1e-9, device="cpu"),
    )
    assert d.status == CONVERGED
    np.testing.assert_allclose(state.coupling, ref.coupling, atol=1e-10)
    assert np.linalg.eigvalsh(state.coupling).min() > -1e-10


def test_quantum_zero_budget_reports_initial_infeasibility():
    cost, a, b = case()
    state, d = solve_quantum_quadratic(cost, a, b, 0.7, 1e-9, 0)
    assert state.iterations == 0
    assert d.status != CONVERGED
    assert np.isfinite(d.error)
    _, d = solve_quantum_quadratic(cost, a, b, jnp.nan, 1e-9, 1)
    assert d.status == INVALID_INPUT


def test_real_cost_keeps_complex_marginals():
    _, a, b = case()
    cost = jnp.zeros((6, 6))
    state, d = solve_quantum_quadratic(cost, a, b, 0.7, 1e-9, 10000)
    assert d.status == CONVERGED
    from erot.quantum import partial_trace_second

    np.testing.assert_allclose(partial_trace_second(state.coupling, 2, 3), a, atol=1e-9)


def test_quantum_iteration_controls_do_not_wrap_int32():
    cost, a, b = case()
    _, d = solve_quantum_quadratic(cost, a, b, 0.7, 1e-9, jnp.array(2**32, jnp.int64))
    assert d.status == INVALID_INPUT
    state, _ = solve_quantum_quadratic(cost, a, b, 0.7, 1e-9, 0)
    state = state._replace(iterations=jnp.array(2**32, jnp.int64))
    _, d = solve_quantum_quadratic(cost, a, b, 0.7, 1e-9, 1, state=state)
    assert d.status == INVALID_INPUT
