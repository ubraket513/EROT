"""Physical time advances only for successful JKO subproblems."""

import jax
import jax.numpy as jnp
import numpy as np

from erot.flows.functionals import Quadratic
from erot.flows.trajectory import initialize_flow, jko_step, run_flow_chunk
from erot.solvers.state import CONVERGED

jax.config.update("jax_enable_x64", True)


def case():
    return (
        jnp.array([[0.0, 0.3], [0.3, 0.0]]),
        jnp.array([0.3, 0.7]),
        Quadratic(jnp.array([0.6, 0.4])),
    )


def test_failed_step_keeps_mass_time_and_backend_resume_state():
    c, p, e = case()
    initial = initialize_flow(p, backend="pdhg")
    failed, d = jko_step(initial, c, e, 0.3, 1e-10, 1, backend="pdhg")
    assert d.status != CONVERGED
    np.testing.assert_array_equal(failed.rho, p)
    assert failed.time == 0 and failed.accepted_steps == 0
    assert failed.attempted_steps == 1
    assert failed.solver_state.iterations == 1
    accepted, d = jko_step(failed, c, e, 0.3, 1e-8, 10000, backend="pdhg")
    assert d.status == CONVERGED
    assert accepted.time == 0.3 and accepted.accepted_steps == 1
    assert accepted.attempted_steps == 2
    assert abs(accepted.rho.sum() - p.sum()) < 1e-8


def test_entropic_failure_does_not_advance_time():
    c, p, e = case()
    initial = initialize_flow(p, backend="sinkhorn")
    failed, d = jko_step(
        initial,
        c,
        e,
        0.3,
        1e-8,
        20,
        backend="sinkhorn",
        epsilon=0.01,
        inner_iterations=0,
    )
    assert d.status != CONVERGED and failed.time == 0
    np.testing.assert_array_equal(failed.rho, p)


def test_selective_chunk_matches_manual_steps_and_can_compile():
    c, p, e = case()
    initial = initialize_flow(p, backend="sinkhorn")
    options = dict(backend="sinkhorn", epsilon=0.2)
    run = jax.jit(
        lambda s: run_flow_chunk(
            s, c, e, 0.3, 1e-7, 200, steps=3, snapshot_stride=2, **options
        )
    )
    result = run(initial)
    manual = initial
    for _ in range(3):
        manual, d = jko_step(manual, c, e, 0.3, 1e-7, 200, **options)
        assert d.status == CONVERGED
    np.testing.assert_allclose(result.state.rho, manual.rho, atol=1e-10)
    assert result.state.accepted_steps == 3
    assert result.snapshots.shape == (
        2,
        2,
    )  # Initial plus after step 2, not every step.
    np.testing.assert_allclose(result.snapshot_times, [0.0, 0.6], atol=1e-12)
    assert result.state.time == 0.3 * 3


def test_chunk_stops_attempting_after_failure():
    c, p, e = case()
    initial = initialize_flow(p, backend="pdhg")
    result = run_flow_chunk(
        initial, c, e, 0.3, 1e-12, 1, steps=4, backend="pdhg", snapshot_stride=2
    )
    assert result.state.attempted_steps == 1
    assert result.state.time == 0
    assert result.diagnostics.status != CONVERGED
