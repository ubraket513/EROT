# SPDX-License-Identifier: Apache-2.0
"""Primal-dual hybrid gradient for unregularized discrete JKO steps."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from ..solvers.state import CONVERGED, INVALID_INPUT, ITERATION_LIMIT, NUMERICAL_FAILURE
from .functionals import Energy
from .state import JKODiagnostics, PDHGState


def coupled_operator_norm_squared(n: int, m: int) -> float:
    """Exact norm² of K(pi,rho)=(rows(pi)-rho, columns(pi)).

    KK* has diagonal blocks (m+1)I_n, nI_m and all-ones off-diagonal
    blocks. Its largest eigenvalue is that of the uniform-mode 2x2 block.
    """
    if n <= 0 or m <= 0:
        raise ValueError("grid sizes must be positive")
    return (m + n + 1 + math.sqrt((m + 1 - n) ** 2 + 4 * n * m)) / 2


def solve_pdhg_jko(
    cost: jax.Array,
    previous: jax.Array,
    energy: Energy,
    time_step: jax.Array,
    tolerance: jax.Array,
    max_iterations: jax.Array,
    *,
    primal_step: jax.Array | None = None,
    dual_step: jax.Array | None = None,
    theta: jax.Array = 1.0,
    state: PDHGState | None = None,
) -> tuple[PDHGState, JKODiagnostics]:
    """Minimize dt*F(rho)+0.5*<C,pi> with nonnegative pi and prescribed columns.

    Additional work budget and both step sizes are explicit. The default steps
    satisfy tau*sigma*||K||² < 1. Inputs are real, finite cell masses with positive
    total mass. Resume requires identical problem and step parameters, including
    theta. Feasibility and proximal fixed-point stationarity both gate success.
    """
    cost, previous = jnp.asarray(cost), jnp.asarray(previous)
    if cost.ndim != 2 or previous.shape != (cost.shape[1],):
        raise ValueError("cost must be a matrix with one column per previous mass")
    if not jnp.issubdtype(cost.dtype, jnp.floating) or jnp.iscomplexobj(previous):
        raise ValueError("PDHG requires real floating costs and real masses")
    previous = previous.astype(cost.dtype)
    n, m = cost.shape
    norm2 = coupled_operator_norm_squared(n, m)
    default_step = 0.99 / math.sqrt(norm2)
    tau = jnp.asarray(default_step if primal_step is None else primal_step, cost.dtype)
    sigma = jnp.asarray(default_step if dual_step is None else dual_step, cost.dtype)
    dt, tol, theta = (jnp.asarray(x, cost.dtype) for x in (time_step, tolerance, theta))
    budget = jnp.asarray(max_iterations)
    mass = previous.sum()
    valid = (
        jnp.isfinite(cost).all()
        & jnp.isfinite(previous).all()
        & (previous >= 0).all()
        & (mass > 0)
        & energy.is_valid()
        & jnp.isfinite(dt)
        & (dt > 0)
        & jnp.isfinite(tol)
        & (tol > 0)
        & jnp.isfinite(tau)
        & (tau > 0)
        & jnp.isfinite(sigma)
        & (sigma > 0)
        & (tau * sigma * norm2 < 1)
        & jnp.isfinite(theta)
        & (theta >= 0)
        & (theta <= 1)
        & jnp.isfinite(budget)
        & (budget >= 0)
        & (budget == jnp.floor(budget))
    )
    if state is None:
        rho = jnp.full((n,), mass / n, dtype=cost.dtype)
        coupling = rho[:, None] * previous[None, :] / mass
        state = PDHGState(
            coupling,
            rho,
            jnp.zeros(n, cost.dtype),
            jnp.zeros(m, cost.dtype),
            coupling,
            rho,
            jnp.asarray(0, jnp.int32),
        )
    else:
        expected = ((n, m), (n,), (n,), (m,), (n, m), (n,))
        if any(x.shape != shape for x, shape in zip(state[:-1], expected, strict=True)):
            raise ValueError("PDHG resume shapes do not match this problem")
        for value in state[:-1]:
            valid = valid & jnp.isfinite(value).all()
    count = jnp.asarray(state.iterations)
    valid = valid & jnp.isfinite(count) & (count >= 0) & (count == jnp.floor(count))
    valid = valid & (count <= 2**31 - 1) & (budget <= 2**31 - 1 - count)
    stop = count.astype(jnp.int32) + budget.astype(jnp.int32)
    state = state._replace(iterations=count.astype(jnp.int32))

    def metrics(current):
        pi, rho, u, v, _, _, _ = current
        row = jnp.max(jnp.abs(pi.sum(axis=1) - rho))
        column = jnp.max(jnp.abs(pi.sum(axis=0) - previous))
        positivity = jnp.maximum(0.0, jnp.maximum(-pi.min(), -rho.min()))
        feasibility = jnp.maximum(jnp.maximum(row, column), positivity)
        pi_fixed = pi - jnp.maximum(pi - (0.5 * cost + u[:, None] + v[None, :]), 0.0)
        rho_fixed = rho - energy.prox(rho + u, dt)
        stationarity = jnp.maximum(
            jnp.max(jnp.abs(pi_fixed)), jnp.max(jnp.abs(rho_fixed))
        )
        objective = dt * energy.value(rho) + 0.5 * jnp.sum(cost * pi)
        error = jnp.where(
            jnp.isfinite(objective), jnp.maximum(feasibility, stationarity), jnp.inf
        )
        return objective, feasibility, stationarity, error

    def condition(carry):
        current, info = carry
        return (
            valid
            & jnp.isfinite(info[-1])
            & (current.iterations < stop)
            & (info[-1] > tol)
        )

    def body(carry):
        current, _ = carry
        pi, rho, u, v, bar_pi, bar_rho, iteration = current
        # Dual-first Chambolle-Pock; extrapolated primal belongs to this state.
        new_u = u + sigma * (bar_pi.sum(axis=1) - bar_rho)
        new_v = v + sigma * (bar_pi.sum(axis=0) - previous)
        new_pi = jnp.maximum(
            pi - tau * (0.5 * cost + new_u[:, None] + new_v[None, :]), 0.0
        )
        new_rho = energy.prox(rho + tau * new_u, tau * dt)
        updated = PDHGState(
            new_pi,
            new_rho,
            new_u,
            new_v,
            new_pi + theta * (new_pi - pi),
            new_rho + theta * (new_rho - rho),
            iteration + 1,
        )
        return updated, metrics(updated)

    final, info = jax.lax.while_loop(condition, body, (state, metrics(state)))
    objective, feasibility, stationarity, error = info
    status = jnp.where(
        ~valid,
        INVALID_INPUT,
        jnp.where(
            ~jnp.isfinite(error),
            NUMERICAL_FAILURE,
            jnp.where(error <= tol, CONVERGED, ITERATION_LIMIT),
        ),
    )
    return final, JKODiagnostics(
        objective, feasibility, stationarity, error, final.iterations, status
    )
