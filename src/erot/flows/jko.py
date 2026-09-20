# SPDX-License-Identifier: Apache-2.0
"""Finite-epsilon JKO density optimization using shared Sinkhorn potentials."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import xlogy

from ..solvers import SinkhornWarmStart, materialize_plan, solve_sinkhorn
from ..solvers.state import CONVERGED, INVALID_INPUT, ITERATION_LIMIT, NUMERICAL_FAILURE
from .functionals import Energy
from .state import (
    INNER_SOLVE_FAILED,
    LINE_SEARCH_FAILED,
    EntropicJKOState,
    JKODiagnostics,
)


def _simplex_projection(values, mass):
    ordered = jnp.sort(values)[::-1]
    shift = (jnp.cumsum(ordered) - mass) / jnp.arange(
        1, values.size + 1, dtype=values.dtype
    )
    index = jnp.maximum(jnp.sum(ordered > shift) - 1, 0)
    return jnp.maximum(values - shift[index], 0.0)


def solve_entropic_jko(
    cost: jax.Array,
    previous: jax.Array,
    energy: Energy,
    time_step: jax.Array,
    epsilon: jax.Array,
    tolerance: jax.Array,
    max_iterations: jax.Array,
    *,
    inner_tolerance: jax.Array = 1e-10,
    inner_iterations: jax.Array = 10000,
    learning_rate: jax.Array = 1.0,
    method: str = "mirror",
    state: EntropicJKOState | None = None,
) -> tuple[EntropicJKOState, JKODiagnostics]:
    """Minimize F(rho)+T_epsilon(rho,previous)/(2*dt) at fixed positive mass.

    Mirror descent (default) or projected gradient (method='sgd') uses Armijo
    backtracking and f/(2*dt) from cost-unit Sinkhorn potentials. Method is static
    under jit. An unconverged inner transport solve is an explicit failure.
    Resume requires the identical problem and numerical controls. This dense
    implementation materializes plans for objective evaluation.
    """
    if method not in ("mirror", "sgd"):
        raise ValueError("method must be 'mirror' or 'sgd'")
    cost, previous = jnp.asarray(cost), jnp.asarray(previous)
    if cost.ndim != 2 or previous.shape != (cost.shape[1],):
        raise ValueError("cost columns must match the previous mass vector")
    if not jnp.issubdtype(cost.dtype, jnp.floating) or jnp.iscomplexobj(previous):
        raise ValueError("entropic JKO requires real floating cost and real masses")
    previous = previous.astype(cost.dtype)
    dt, eps, tol, inner_tol, rate = (
        jnp.asarray(x, cost.dtype)
        for x in (time_step, epsilon, tolerance, inner_tolerance, learning_rate)
    )
    budget, inner_budget = jnp.asarray(max_iterations), jnp.asarray(inner_iterations)
    mass = previous.sum()
    valid = (
        jnp.isfinite(cost).all()
        & jnp.isfinite(previous).all()
        & (previous >= 0).all()
        & (mass > 0)
        & energy.is_valid()
    )
    for value in (dt, eps, tol, inner_tol, rate):
        valid = valid & jnp.isfinite(value) & (value > 0)
    valid = valid & (inner_tol <= tol)
    for value in (budget, inner_budget):
        valid = (
            valid
            & jnp.isfinite(value)
            & (value >= 0)
            & (value <= 2**31 - 1)
            & (value == jnp.floor(value))
        )
    if state is None:
        rho = jnp.full((cost.shape[0],), mass / cost.shape[0], cost.dtype)
        state = EntropicJKOState(
            rho,
            (jnp.zeros_like(rho), jnp.zeros_like(previous)),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int32),
            rate,
        )
    else:
        if state.rho.shape != (cost.shape[0],) or tuple(
            x.shape for x in state.potentials
        ) != (state.rho.shape, previous.shape):
            raise ValueError("entropic JKO resume state has incompatible shapes")
        valid = valid & jnp.isfinite(state.rho).all() & (state.rho > 0).all()
        valid = valid & jnp.isclose(
            state.rho.sum(), mass, rtol=1e-6 if cost.dtype == jnp.float32 else 1e-10
        )
        valid = valid & jnp.isfinite(state.step_size) & (state.step_size > 0)
    count = jnp.asarray(state.iterations)
    valid = (
        valid
        & jnp.isfinite(count)
        & (count >= 0)
        & (count == jnp.floor(count))
        & (count <= 2**31 - 1 - budget)
    )
    stop = count.astype(jnp.int32) + budget.astype(jnp.int32)
    state = state._replace(iterations=count.astype(jnp.int32))
    work_count = jnp.asarray(state.inner_iterations)
    valid = (
        valid
        & jnp.isfinite(work_count)
        & (work_count >= 0)
        & (work_count <= 2**31 - 1)
        & (work_count == jnp.floor(work_count))
    )
    state = state._replace(inner_iterations=work_count.astype(jnp.int32))

    def add_work(total, additional):
        fits = additional <= (2**31 - 1 - total)
        return jnp.where(fits, total + additional, total), fits

    def evaluate(rho, potentials):
        transport, diag = solve_sinkhorn(
            cost,
            (rho, previous),
            eps,
            inner_tol,
            inner_budget,
            warm_start=SinkhornWarmStart(potentials),
        )
        plan = materialize_plan(cost, transport.potentials, eps)
        ot = jnp.sum(cost * plan) + eps * jnp.sum(xlogy(plan, plan) - plan)
        objective = energy.value(rho) + ot / (2 * dt)
        dual_ot = (
            jnp.dot(rho, transport.potentials[0])
            + jnp.sum(previous * jnp.where(previous > 0, transport.potentials[1], 0.0))
            - eps * jnp.sum(plan)
        )
        descent_objective = energy.value(rho) + dual_ot / (2 * dt)
        gradient = energy.gradient(rho) + transport.potentials[0] / (2 * dt)
        centered = gradient - jnp.sum(rho * gradient) / mass
        stationarity = jnp.max(jnp.abs(centered))
        feasibility = jnp.maximum(diag.error, jnp.abs(rho.sum() - mass))
        finite = jnp.isfinite(objective) & jnp.isfinite(gradient).all()
        error = jnp.where(finite, jnp.maximum(stationarity, feasibility), jnp.inf)
        return (
            transport.potentials,
            (
                objective,
                feasibility,
                stationarity,
                error,
                gradient,
                diag.status,
                descent_objective,
            ),
            diag.iterations,
        )

    potentials, metrics, work = evaluate(state.rho, state.potentials)
    total_work, fits = add_work(state.inner_iterations, work)
    state = state._replace(potentials=potentials, inner_iterations=total_work)
    initial_failure = jnp.where(metrics[5] != CONVERGED, INNER_SOLVE_FAILED, CONVERGED)
    initial_failure = jnp.where(fits, initial_failure, NUMERICAL_FAILURE)

    def condition(carry):
        current, info, failure = carry
        return (
            valid
            & (failure == CONVERGED)
            & jnp.isfinite(info[3])
            & (info[3] > tol)
            & (current.iterations < stop)
        )

    def step(carry):
        current, info, _ = carry
        gradient = info[4]

        def line_condition(line):
            attempt, _, _, _, _, accepted, failure, _ = line
            return (attempt < 30) & ~accepted & (failure == CONVERGED)

        def line_step(line):
            attempt, step_size, _, _, _, _, _, total_work = line
            if method == "mirror":
                proposal = mass * jax.nn.softmax(
                    jnp.log(current.rho) - step_size * gradient
                )
            else:
                proposal = _simplex_projection(current.rho - step_size * gradient, mass)
            proposal = jnp.maximum(proposal, jnp.finfo(cost.dtype).tiny)
            proposal = proposal * (mass / proposal.sum())
            new_potentials, new_info, work = evaluate(proposal, current.potentials)
            total_work, fits = add_work(total_work, work)
            finite = jnp.isfinite(new_info[3])
            inner_ok = new_info[5] == CONVERGED
            roundoff = 32 * jnp.finfo(cost.dtype).eps * (1 + jnp.abs(info[6]))
            accepted = (
                finite
                & inner_ok
                & fits
                & (
                    new_info[6]
                    <= info[6]
                    + 1e-4 * jnp.dot(gradient, proposal - current.rho)
                    + roundoff
                )
            )
            failure = jnp.where(~inner_ok, INNER_SOLVE_FAILED, CONVERGED)
            failure = jnp.where(fits, failure, NUMERICAL_FAILURE)
            return (
                attempt + 1,
                jnp.where(accepted, step_size, step_size / 2),
                proposal,
                new_potentials,
                new_info,
                accepted,
                failure,
                total_work,
            )

        line = (
            jnp.asarray(0, jnp.int32),
            current.step_size,
            current.rho,
            current.potentials,
            info,
            jnp.asarray(False),
            jnp.asarray(CONVERGED),
            current.inner_iterations,
        )
        _, rate, rho, potentials, new_info, accepted, failure, work = (
            jax.lax.while_loop(line_condition, line_step, line)
        )
        candidate = EntropicJKOState(
            rho,
            potentials,
            current.iterations + 1,
            work,
            rate,
        )
        retained = current._replace(inner_iterations=work)
        updated = jax.tree.map(
            lambda new, old: jnp.where(accepted, new, old), candidate, retained
        )
        next_info = jax.tree.map(
            lambda new, old: jnp.where(accepted, new, old), new_info, info
        )
        failure = jnp.where(
            accepted,
            CONVERGED,
            jnp.where(failure != CONVERGED, failure, LINE_SEARCH_FAILED),
        )
        return updated, next_info, failure

    final, info, failure = jax.lax.while_loop(
        condition, step, (state, metrics, initial_failure)
    )
    status = jnp.where(
        ~valid,
        INVALID_INPUT,
        jnp.where(
            failure != CONVERGED,
            failure,
            jnp.where(
                ~jnp.isfinite(info[3]),
                NUMERICAL_FAILURE,
                jnp.where(info[3] <= tol, CONVERGED, ITERATION_LIMIT),
            ),
        ),
    )
    return final, JKODiagnostics(
        info[0], info[1], info[2], info[3], final.iterations, status
    )
