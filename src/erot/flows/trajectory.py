# SPDX-License-Identifier: Apache-2.0
"""Acceptance, physical time and selective snapshots for discrete JKO flows."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..solvers.state import CONVERGED, INVALID_INPUT, ITERATION_LIMIT
from .functionals import Energy
from .jko import solve_entropic_jko
from .pdhg import solve_pdhg_jko
from .state import EntropicJKOState, JKODiagnostics, PDHGState


class FlowState(NamedTuple):
    """Accepted physical state plus latest complete (possibly failed) inner state."""

    rho: jax.Array
    time: jax.Array
    accepted_steps: jax.Array
    attempted_steps: jax.Array
    work: jax.Array
    status: jax.Array
    solver_state: PDHGState | EntropicJKOState


class FlowChunk(NamedTuple):
    """Final continuation state, selected mass snapshots and their physical times."""

    state: FlowState
    snapshots: jax.Array
    snapshot_times: jax.Array
    diagnostics: JKODiagnostics


def initialize_flow(
    rho: jax.Array, *, backend: str = "sinkhorn", learning_rate=1.0
) -> FlowState:
    """Prepare state without solving; configure JAX precision/placement beforehand."""
    rho = jnp.asarray(rho)
    if rho.ndim != 1 or rho.size == 0 or not jnp.issubdtype(rho.dtype, jnp.floating):
        raise ValueError("initial flow masses must be a nonempty real floating vector")
    zero = jnp.asarray(0, jnp.int32)
    trial = jnp.full_like(rho, rho.sum() / rho.size)
    if backend == "sinkhorn":
        inner = EntropicJKOState(
            trial,
            (jnp.zeros_like(rho), jnp.zeros_like(rho)),
            zero,
            zero,
            jnp.asarray(learning_rate, rho.dtype),
        )
    elif backend == "pdhg":
        coupling = trial[:, None] * rho[None, :] / rho.sum()
        inner = PDHGState(
            coupling,
            trial,
            jnp.zeros_like(rho),
            jnp.zeros_like(rho),
            coupling,
            trial,
            zero,
        )
    else:
        raise ValueError("backend must be 'sinkhorn' or 'pdhg'")
    return FlowState(
        rho,
        jnp.asarray(0.0, rho.dtype),
        zero,
        zero,
        zero,
        jnp.asarray(CONVERGED),
        inner,
    )


def jko_step(
    state: FlowState,
    cost: jax.Array,
    energy: Energy,
    time_step: jax.Array,
    tolerance: jax.Array,
    max_iterations: jax.Array,
    *,
    backend: str = "sinkhorn",
    epsilon: jax.Array = 0.1,
    **solver_options,
) -> tuple[FlowState, JKODiagnostics]:
    """Attempt one physical step; failed steps retain density/time and resume data.

    Retrying a failed step resumes the same subproblem. A step after success
    initializes a fresh subproblem, because its previous density has changed.
    Changing problem inputs during a failed-step retry requires reinitializing
    the flow. Backend and solver method are structural choices under jit.
    """
    cost = jnp.asarray(cost)
    if cost.shape != (state.rho.size, state.rho.size):
        raise ValueError(
            "a trajectory requires a square cost matching its fixed mass grid"
        )
    if backend == "sinkhorn":
        expected = EntropicJKOState
    elif backend == "pdhg":
        expected = PDHGState
    else:
        raise ValueError("backend must be 'sinkhorn' or 'pdhg'")
    if not isinstance(state.solver_state, expected):
        raise ValueError("flow state backend does not match the requested solver")
    fresh = initialize_flow(
        state.rho,
        backend=backend,
        learning_rate=solver_options.get("learning_rate", 1.0),
    ).solver_state
    pending = (state.status != CONVERGED) & (state.attempted_steps > 0)
    initial = jax.tree.map(
        lambda old, new: jnp.where(pending, old, new), state.solver_state, fresh
    )
    if backend == "sinkhorn":
        solved, diagnostics = solve_entropic_jko(
            cost,
            state.rho,
            energy,
            time_step,
            epsilon,
            tolerance,
            max_iterations,
            state=initial,
            **solver_options,
        )
        additional = solved.inner_iterations - initial.inner_iterations
    else:
        solved, diagnostics = solve_pdhg_jko(
            cost,
            state.rho,
            energy,
            time_step,
            tolerance,
            max_iterations,
            state=initial,
            **solver_options,
        )
        additional = solved.iterations - initial.iterations
    valid = (
        jnp.isfinite(state.time)
        & (state.time >= 0)
        & jnp.isfinite(state.time + time_step)
        & (state.accepted_steps >= 0)
        & (state.accepted_steps < 2**31 - 1)
        & (state.attempted_steps >= state.accepted_steps)
        & (state.attempted_steps < 2**31 - 1)
        & (state.work >= 0)
        & (additional >= 0)
        & (additional <= 2**31 - 1 - state.work)
    )
    status = jnp.where(valid, diagnostics.status, INVALID_INPUT)
    accepted = status == CONVERGED
    result = FlowState(
        jnp.where(accepted, solved.rho, state.rho),
        jnp.where(accepted, state.time + time_step, state.time),
        state.accepted_steps + accepted.astype(jnp.int32),
        state.attempted_steps + valid.astype(jnp.int32),
        jnp.where(valid, state.work + additional, state.work),
        status,
        solved,
    )
    return result, diagnostics._replace(status=status)


def run_flow_chunk(
    state: FlowState,
    cost: jax.Array,
    energy: Energy,
    time_step: jax.Array,
    tolerance: jax.Array,
    max_iterations: jax.Array,
    *,
    steps: int,
    snapshot_stride: int | None = 1,
    **solver_options,
) -> FlowChunk:
    """Run up to steps attempts, stopping at failure without advancing time.

    steps/stride are static. Allocate only requested snapshots: initial plus
    every stride attempts, or none for stride=None. Final state is always
    returned. Repeated times after failure explicitly denote unchanged state.
    An explicit new chunk may retry a pending failed step once.
    """
    if not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative static integer")
    if snapshot_stride is not None and (
        not isinstance(snapshot_stride, int) or snapshot_stride <= 0
    ):
        raise ValueError("snapshot_stride must be a positive static integer or None")
    size = 0 if snapshot_stride is None else 1 + steps // snapshot_stride
    snapshots = jnp.zeros((size, state.rho.size), state.rho.dtype)
    times = jnp.zeros((size,), state.time.dtype)
    if size:
        snapshots = snapshots.at[0].set(state.rho)
        times = times.at[0].set(state.time)
    inf = jnp.asarray(jnp.inf, state.rho.dtype)
    initial_diagnostics = JKODiagnostics(
        jnp.asarray(0.0, state.rho.dtype),
        inf,
        inf,
        inf,
        jnp.asarray(0, jnp.int32),
        jnp.asarray(ITERATION_LIMIT),
    )

    def body(index, carry):
        current, diagnostics, snapshots, times = carry
        current, diagnostics = jax.lax.cond(
            (index == 0) | (current.status == CONVERGED),
            lambda _: jko_step(
                current,
                cost,
                energy,
                time_step,
                tolerance,
                max_iterations,
                **solver_options,
            ),
            lambda _: (current, diagnostics),
            None,
        )
        if snapshot_stride is not None:

            def record(arrays):
                snapshots, times = arrays
                slot = (index + 1) // snapshot_stride
                return snapshots.at[slot].set(current.rho), times.at[slot].set(
                    current.time
                )

            snapshots, times = jax.lax.cond(
                (index + 1) % snapshot_stride == 0,
                record,
                lambda x: x,
                (snapshots, times),
            )
        return current, diagnostics, snapshots, times

    final, diagnostics, snapshots, times = jax.lax.fori_loop(
        0, steps, body, (state, initial_diagnostics, snapshots, times)
    )
    return FlowChunk(final, snapshots, times, diagnostics)
