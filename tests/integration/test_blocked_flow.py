"""Entropic trajectories use implicit geometry without a dense-plan fallback."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.benchmark_blocked_sinkhorn import compiled_memory
from erot.flows import Quadratic, initialize_flow, run_flow_chunk
from erot.geometry import PointCloudGeometry

jax.config.update("jax_enable_x64", True)


def test_blocked_flow_matches_dense_accepted_trajectory():
    x = jnp.linspace(0, 1, 5)[:, None]
    geom = PointCloudGeometry(x, x)
    cost = (x - x.T) ** 2
    rho = jnp.array([0.1, 0.2, 0.4, 0.2, 0.1])
    energy = Quadratic(jnp.ones(5) / 5, 2.0)
    initial = initialize_flow(rho)
    run = jax.jit(
        lambda c: run_flow_chunk(
            initial,
            c,
            energy,
            0.3,
            1e-6,
            1000,
            epsilon=0.2,
            steps=2,
            transport_block_size=3,
        )
    )
    blocked, dense = run(geom), run(cost)
    assert blocked.state.status == dense.state.status == 0
    assert blocked.state.accepted_steps == dense.state.accepted_steps == 2
    np.testing.assert_allclose(blocked.snapshots, dense.snapshots, atol=2e-6)
    np.testing.assert_allclose(
        blocked.diagnostics.objective, dense.diagnostics.objective, atol=1e-7
    )


def test_blocked_inner_failure_keeps_time_and_pdhg_requires_dense():
    x = jnp.linspace(0, 1, 5)[:, None]
    geom = PointCloudGeometry(x, x)
    rho = jnp.array([0.1, 0.2, 0.4, 0.2, 0.1])
    energy = Quadratic(jnp.ones(5) / 5)
    chunk = run_flow_chunk(
        initialize_flow(rho),
        geom,
        energy,
        0.3,
        1e-8,
        100,
        epsilon=0.01,
        steps=2,
        inner_iterations=0,
        transport_block_size=3,
    )
    assert chunk.state.status != 0 and chunk.state.time == 0
    np.testing.assert_array_equal(chunk.state.rho, rho)
    with pytest.raises(ValueError, match="PDHG requires"):
        run_flow_chunk(
            initialize_flow(rho, backend="pdhg"),
            geom,
            energy,
            0.3,
            1e-8,
            100,
            steps=1,
            backend="pdhg",
        )


def test_compiled_flow_has_no_dense_cost_or_coupling():
    n = 129
    x = jnp.linspace(0, 1, n)[:, None]
    geom = PointCloudGeometry(x, x)
    rho = jnp.ones(n) / n
    energy = Quadratic(rho)
    function = jax.jit(
        lambda s, g, e: run_flow_chunk(
            s,
            g,
            e,
            0.3,
            1e-6,
            100,
            epsilon=0.2,
            steps=2,
            snapshot_stride=None,
            transport_block_size=16,
        )
    )
    compiled = function.lower(initialize_flow(rho), geom, energy).compile()
    assert compiled_memory(compiled, n, n)["full_matrix_shapes"] == []
