"""Matched dense references for the implicit point-cloud path."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from erot.geometry import DenseGeometry, PointCloudGeometry
from erot.geometry.plan import apply_transport, plan_block, transport_objective
from erot.solvers.blocked_sinkhorn import solve_blocked_sinkhorn
from erot.solvers.sinkhorn import materialize_plan, solve_sinkhorn
from erot.solvers.state import INVALID_INPUT, SinkhornWarmStart

jax.config.update("jax_enable_x64", True)


def case():
    x = jnp.linspace(0, 1, 5)[:, None]
    y = jnp.linspace(0, 1, 7)[:, None]
    a = jnp.array([0.1, 0.0, 0.2, 0.4, 0.3]) * 2
    b = jnp.array([0.0, 0.0, 0.0, 0.0, 0.2, 0.3, 0.5]) * 2
    return PointCloudGeometry(x, y), (a, b), (x - y.T) ** 2


@pytest.mark.parametrize("epsilon", [0.3, 0.01])
def test_dense_equivalence_objective_blocks_and_application(epsilon):
    geometry, marginals, cost = case()
    dense, dd = solve_sinkhorn(cost, marginals, epsilon, 1e-9, 20000)
    expected = materialize_plan(cost, dense.potentials, epsilon)
    for geom in (geometry, DenseGeometry(cost)):
        state, d = jax.jit(
            lambda g: solve_blocked_sinkhorn(
                g, marginals, epsilon, 1e-9, 20000, block_size=4
            )
        )(geom)
        assert d.status == dd.status == 0
        actual = np.zeros((8, 8))
        for i in (0, 4):
            for j in (0, 4):
                actual[i : i + 4, j : j + 4] = plan_block(
                    geom, state.potentials, epsilon, i, j, block_size=4
                )
        np.testing.assert_allclose(actual[:5, :7], expected, atol=3e-9)
        assert np.count_nonzero(actual[5:]) == np.count_nonzero(actual[:, 7:]) == 0
        positive = np.asarray(expected) > 0
        objective = np.sum(np.asarray(expected) * np.asarray(cost))
        objective += epsilon * np.sum(
            np.asarray(expected)[positive]
            * (np.log(np.asarray(expected)[positive]) - 1)
        )
        np.testing.assert_allclose(
            transport_objective(geom, state.potentials, epsilon, block_size=4),
            objective,
            atol=1e-8,
        )
        values = jnp.arange(21.0).reshape(7, 3) - 10
        np.testing.assert_allclose(
            apply_transport(geom, state.potentials, epsilon, values, block_size=4),
            expected @ values,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            apply_transport(
                geom, state.potentials, epsilon, values[:, 0], block_size=4
            ),
            expected @ values[:, 0],
            atol=1e-8,
        )


def test_resume_and_warm_start_changed_support():
    geom, marginals, _ = case()
    kwargs = dict(block_size=4)
    first, _ = solve_blocked_sinkhorn(geom, marginals, 0.01, 1e-14, 3, **kwargs)
    resumed, _ = solve_blocked_sinkhorn(
        geom, marginals, 0.01, 1e-14, 7, state=first, **kwargs
    )
    whole, _ = solve_blocked_sinkhorn(geom, marginals, 0.01, 1e-14, 10, **kwargs)
    for a, b in zip(jax.tree.leaves(resumed), jax.tree.leaves(whole)):
        np.testing.assert_array_equal(a, b)
    changed = (jnp.ones(5) / 5, jnp.ones(7) / 7)
    warm, d = solve_blocked_sinkhorn(
        geom,
        changed,
        0.2,
        1e-9,
        2000,
        warm_start=SinkhornWarmStart(first.potentials),
        **kwargs,
    )
    assert d.status == 0
    assert all(np.isfinite(p).all() for p in warm.potentials)


@pytest.mark.parametrize("budget", [-1, 1.5, float("nan"), np.float32(2**31)])
def test_invalid_budget(budget):
    geom, marginals, _ = case()
    _, d = solve_blocked_sinkhorn(geom, marginals, 0.3, 1e-8, budget, block_size=4)
    assert d.status == INVALID_INPUT


def test_zero_budget_bad_mass_and_invalid_geometry():
    geom, marginals, _ = case()
    state, d = solve_blocked_sinkhorn(geom, marginals, 0.3, 1e-10, 0, block_size=4)
    assert state.iterations == 0 and d.status == 1
    _, d = solve_blocked_sinkhorn(
        geom, (marginals[0] * 2, marginals[1]), 0.3, 1e-8, 10, block_size=4
    )
    assert d.status == INVALID_INPUT
    bad = geom._replace(x=geom.x.at[0, 0].set(jnp.nan))
    _, d = solve_blocked_sinkhorn(bad, marginals, 0.3, 1e-8, 10, block_size=4)
    assert d.status == INVALID_INPUT


def test_tiny_epsilon_jit_batch_and_checked_resume_counter():
    x = jnp.arange(5.0)[:, None]
    geom = PointCloudGeometry(x, x)
    marginal = jnp.ones(5) / 5
    solve = jax.jit(
        jax.vmap(
            lambda eps: solve_blocked_sinkhorn(
                geom, (marginal, marginal), eps, 1e-10, 100, block_size=3
            )
        )
    )
    states, diagnostics = solve(jnp.array([1e-4, 1e-6]))
    np.testing.assert_array_equal(diagnostics.status, [0, 0])
    np.testing.assert_allclose(
        jnp.exp(states.potentials[0] / jnp.array([1e-4, 1e-6])[:, None]),
        jnp.ones((2, 5)),
        atol=1e-10,
    )
    state, _ = solve_blocked_sinkhorn(
        geom, (marginal, marginal), 0.1, 1e-10, 0, block_size=3
    )
    state = state._replace(iterations=jnp.int64(2**31))
    _, d = solve_blocked_sinkhorn(
        geom,
        (marginal, marginal),
        0.1,
        1e-10,
        jnp.float32(0),
        state=state,
        block_size=3,
    )
    assert d.status == INVALID_INPUT


def test_output_reductions_promote_explicit_potential_precision():
    geom = PointCloudGeometry(
        jnp.array([[0.0], [1.0]], dtype=jnp.float32),
        jnp.array([[0.0], [1.0]], dtype=jnp.float32),
    )
    potentials = (jnp.zeros(2, dtype=jnp.float64), jnp.zeros(2, dtype=jnp.float64))
    expected = np.exp(-np.array([[0.0, 1.0], [1.0, 0.0]]))
    objective = transport_objective(geom, potentials, jnp.float64(1), block_size=3)
    np.testing.assert_allclose(objective, -expected.sum(), atol=1e-12)


@pytest.mark.parametrize("kind", ["dense", "pointcloud"])
def test_numpy_geometry_has_same_eager_and_compiled_contract(kind):
    points = np.array([[0.0], [1.0]])
    geom = (
        DenseGeometry((points - points.T) ** 2)
        if kind == "dense"
        else PointCloudGeometry(points, points)
    )
    a = jnp.ones(2) / 2
    direct, dd = solve_blocked_sinkhorn(geom, (a, a), 0.3, 1e-10, 100, block_size=3)
    compiled, cd = jax.jit(
        lambda g: solve_blocked_sinkhorn(g, (a, a), 0.3, 1e-10, 100, block_size=3)
    )(geom)
    assert dd.status == cd.status == 0
    for left, right in zip(direct.potentials, compiled.potentials):
        np.testing.assert_allclose(left, right, atol=1e-12)
