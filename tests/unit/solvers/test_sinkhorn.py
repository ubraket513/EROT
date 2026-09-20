"""Pure solver state must preserve numerical and transformation contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from erot.solvers import SinkhornWarmStart, materialize_plan, solve_sinkhorn
from erot.solvers.state import CONVERGED, INVALID_INPUT, ITERATION_LIMIT

jax.config.update("jax_enable_x64", True)


def case():
    return jnp.array([[0.0, 0.7], [0.3, 0.1], [0.9, 0.2]]), (
        jnp.array([0.2, 0.5, 0.3]),
        jnp.array([0.4, 0.6]),
    )


def test_dense_wrapper_agreement_and_cost_unit_potentials():
    import erot

    cost, marginals = case()
    state, diagnostics = solve_sinkhorn(cost, marginals, 0.7, 1e-10, 10000)
    plan = materialize_plan(cost, state.potentials, 0.7)
    ref = erot.solve(
        cost,
        marginals,
        problem="classical",
        regularizer="shannon",
        method="sinkhorn",
        config=erot.SolverConfig(0.7, tolerance=1e-10, device="cpu"),
    )
    np.testing.assert_allclose(plan, ref.coupling, atol=1e-10)
    assert diagnostics.status == CONVERGED
    assert abs(jnp.dot(marginals[0], state.potentials[0])) < 1e-12
    np.testing.assert_allclose(
        cost + 0.7 * jnp.log(plan),
        state.potentials[0][:, None] + state.potentials[1][None, :],
        atol=1e-12,
    )
    # Independent optimized-value derivative from Stage 1 CVXPY audit.
    assert float(state.potentials[0][0] - state.potentials[0][1]) == pytest.approx(
        -0.52766123635, abs=1e-5
    )


def test_resume_matches_uninterrupted_sweeps():
    cost, marginals = case()
    first, d = solve_sinkhorn(cost, marginals, 0.07, 1e-15, 2)
    assert d.status == ITERATION_LIMIT
    resumed, rd = solve_sinkhorn(cost, marginals, 0.07, 1e-15, 3, state=first)
    whole, wd = solve_sinkhorn(cost, marginals, 0.07, 1e-15, 5)
    for a, b in zip(resumed.potentials, whole.potentials, strict=True):
        np.testing.assert_array_equal(a, b)
    assert resumed.iterations == whole.iterations == 5
    assert rd.error == wd.error


def test_zero_budget_and_changed_epsilon_warm_start():
    cost, marginals = case()
    zero, diagnostic = solve_sinkhorn(cost, marginals, 0.7, 1e-10, 0)
    assert zero.iterations == 0
    assert diagnostic.status == ITERATION_LIMIT
    state, _ = solve_sinkhorn(cost, marginals, 0.7, 1e-10, 1000)
    warm, d = solve_sinkhorn(
        cost,
        marginals,
        0.2,
        1e-10,
        1000,
        warm_start=SinkhornWarmStart(state.potentials),
    )
    cold, _ = solve_sinkhorn(cost, marginals, 0.2, 1e-10, 1000)
    assert d.status == CONVERGED
    np.testing.assert_allclose(
        materialize_plan(cost, warm.potentials, 0.2),
        materialize_plan(cost, cold.potentials, 0.2),
        atol=2e-10,
    )


def test_multimarginal_zero_support_nonunit_mass():
    marginals = (
        jnp.array([2.0, 0.0]),
        jnp.array([0.5, 1.5]),
        jnp.array([0.0, 1.0, 1.0]),
    )
    cost = jnp.arange(12, dtype=float).reshape(2, 2, 3) / 12
    state, d = solve_sinkhorn(cost, marginals, 0.7, 1e-10, 1000)
    assert d.status == CONVERGED
    plan = materialize_plan(cost, state.potentials, 0.7)
    assert jnp.isfinite(plan).all()
    for axis, marginal in enumerate(marginals):
        np.testing.assert_allclose(
            plan.sum(axis=tuple(i for i in range(3) if i != axis)), marginal, atol=1e-10
        )
    assert jnp.all(plan[1] == 0)
    assert jnp.all(plan[:, :, 0] == 0)


@pytest.mark.parametrize(
    "epsilon,budget", [(float("nan"), 2), (0.0, 2), (0.7, -1), (0.7, 1.5)]
)
def test_invalid_parameters_return_device_status(epsilon, budget):
    cost, marginals = case()
    _, d = jax.jit(solve_sinkhorn)(cost, marginals, epsilon, 1e-10, budget)
    assert d.status == INVALID_INPUT


def test_jit_vmap_and_scan_composition():
    cost, marginals = case()
    batched = jax.jit(jax.vmap(lambda c: solve_sinkhorn(c, marginals, 0.7, 1e-9, 1000)))
    states, diagnostics = batched(jnp.stack([cost, cost * 1.1]))
    assert jnp.all(diagnostics.status == CONVERGED)
    initial = SinkhornWarmStart(tuple(jnp.zeros_like(a) for a in marginals))

    def step(warm, c):
        state, d = solve_sinkhorn(c, marginals, 0.7, 1e-9, 1000, warm_start=warm)
        return SinkhornWarmStart(state.potentials), d

    _, ds = jax.jit(
        lambda: jax.lax.scan(step, initial, jnp.stack([cost, cost * 1.01]))
    )()
    assert jnp.all(ds.status == CONVERGED)
    assert states.potentials[0].shape == (2, 3)


@pytest.mark.parametrize(
    "kind", ["nan_cost", "negative_mass", "unequal_mass", "bad_state"]
)
def test_invalid_problem_and_resume_return_status(kind):
    from erot.solvers import SinkhornState

    cost, marginals = case()
    state = None
    if kind == "nan_cost":
        cost = cost.at[0, 0].set(jnp.nan)
    elif kind == "negative_mass":
        marginals = (jnp.array([-0.1, 0.6, 0.5]), marginals[1])
    elif kind == "unequal_mass":
        marginals = (marginals[0] * 2, marginals[1])
    else:
        state = SinkhornState((jnp.full((3,), jnp.nan), jnp.zeros(2)), jnp.array(0))
    _, diagnostics = jax.jit(solve_sinkhorn)(
        cost, marginals, 0.7, 1e-9, 10, state=state
    )
    assert diagnostics.status == INVALID_INPUT
    assert diagnostics.iterations == 0


def test_warm_start_can_activate_previously_zero_support():
    cost, marginals = case()
    old = (jnp.array([0.2, 0.8, 0.0]), marginals[1])
    previous, _ = solve_sinkhorn(cost, old, 0.7, 1e-10, 1000)
    state, diagnostic = solve_sinkhorn(
        cost,
        marginals,
        0.7,
        1e-10,
        1000,
        warm_start=SinkhornWarmStart(previous.potentials),
    )
    assert diagnostic.status == CONVERGED
    plan = materialize_plan(cost, state.potentials, 0.7)
    np.testing.assert_allclose(plan.sum(axis=1), marginals[0], atol=1e-10)


def test_adjacent_problem_warm_start_work_is_measured():
    cost, marginals = case()
    previous, _ = solve_sinkhorn(cost, marginals, 0.2, 1e-10, 1000)
    adjacent = (marginals[0] + jnp.array([1e-5, -1e-5, 0.0]), marginals[1])
    _, warm = solve_sinkhorn(
        cost,
        adjacent,
        0.2,
        1e-10,
        1000,
        warm_start=SinkhornWarmStart(previous.potentials),
    )
    _, cold = solve_sinkhorn(cost, adjacent, 0.2, 1e-10, 1000)
    assert warm.status == cold.status == CONVERGED
    assert warm.iterations < cold.iterations  # This specific nearby-marginal case.
