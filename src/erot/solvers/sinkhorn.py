"""Composable dense multi-marginal log-domain Sinkhorn without host conversions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from .state import (
    CONVERGED,
    INVALID_INPUT,
    ITERATION_LIMIT,
    NUMERICAL_FAILURE,
    SinkhornState,
    SinkhornWarmStart,
    SolverDiagnostics,
)


def _broadcast(vector, axis, rank):
    shape = [1] * rank
    shape[axis] = vector.shape[0]
    return vector.reshape(shape)


def materialize_plan(
    cost: jax.Array, potentials: tuple[jax.Array, ...], epsilon: jax.Array
) -> jax.Array:
    """Return the dense tensor exp((sum of cost-unit potentials - cost)/epsilon)."""
    exponent = -cost
    for axis, potential in enumerate(potentials):
        exponent = exponent + _broadcast(potential, axis, cost.ndim)
    return jnp.exp(exponent / epsilon)


def _error(cost, marginals, potentials, epsilon):
    coupling = materialize_plan(cost, potentials, epsilon)
    errors = []
    for axis, marginal in enumerate(marginals):
        axes = tuple(i for i in range(cost.ndim) if i != axis)
        errors.append(jnp.sum(jnp.abs(coupling.sum(axis=axes) - marginal)))
    return jnp.max(jnp.stack(errors))


def _gauge(potentials, marginals):
    result = list(potentials)
    for axis in range(len(result) - 1):
        marginal = marginals[axis]
        mean = (
            jnp.sum(marginal * jnp.where(marginal > 0, result[axis], 0))
            / marginal.sum()
        )
        result[axis] = result[axis] - mean
        result[-1] = result[-1] + mean
    return tuple(result)


def solve_sinkhorn(
    cost: jax.Array,
    marginals: tuple[jax.Array, ...],
    epsilon: jax.Array,
    tolerance: jax.Array,
    max_iterations: jax.Array,
    *,
    state: SinkhornState | None = None,
    warm_start: SinkhornWarmStart | None = None,
) -> tuple[SinkhornState, SolverDiagnostics]:
    """Solve dense Shannon OT with device diagnostics and optional reusable state.

    All positive marginals have equal total mass. Scalar controls are dynamic;
    tensor rank/shapes are structural. ``max_iterations`` is additional work,
    including zero. Resume requires identical inputs; use ``warm_start`` when
    epsilon, cost or marginals change. Positive-support potentials are cost-unit
    values; the first k-1 have marginal-weighted mean zero. Supports jit, vmap
    and scan, but does not promise reverse differentiation through the loop.
    """
    cost = jnp.asarray(cost)
    marginals = tuple(jnp.asarray(a, dtype=cost.dtype) for a in marginals)
    if cost.ndim < 2 or len(marginals) != cost.ndim:
        raise ValueError("cost rank must match at least two marginals")
    if not jnp.issubdtype(cost.dtype, jnp.floating):
        raise ValueError("cost must have a real floating dtype")
    if any(a.shape != (cost.shape[i],) for i, a in enumerate(marginals)):
        raise ValueError("marginal lengths must match cost axes")
    if state is not None and warm_start is not None:
        raise ValueError("supply state or warm_start, not both")
    epsilon, tolerance = (
        jnp.asarray(epsilon, cost.dtype),
        jnp.asarray(tolerance, cost.dtype),
    )
    budget = jnp.asarray(max_iterations)
    valid = (
        jnp.isfinite(cost).all()
        & jnp.isfinite(epsilon)
        & (epsilon > 0)
        & jnp.isfinite(tolerance)
        & (tolerance > 0)
        & jnp.isfinite(budget)
        & (budget >= 0)
        & (budget == jnp.floor(budget))
    )
    mass = marginals[0].sum()
    mass_tol = 1e-6 if cost.dtype == jnp.float32 else 1e-10
    for a in marginals:
        valid = valid & jnp.isfinite(a).all() & (a >= 0).all() & (a.sum() > 0)
        valid = valid & jnp.isclose(a.sum(), mass, rtol=mass_tol, atol=mass_tol)
    initial = state if state is not None else warm_start
    if initial is None:
        potentials = tuple(
            jnp.where(a > 0, jnp.zeros_like(a), -jnp.inf) for a in marginals
        )
    else:
        if len(initial.potentials) != len(marginals) or any(
            f.shape != a.shape
            for f, a in zip(initial.potentials, marginals, strict=True)
        ):
            raise ValueError("potential shapes must match marginals")
        potentials = tuple(jnp.asarray(f, cost.dtype) for f in initial.potentials)
        if warm_start is not None:
            potentials = tuple(
                jnp.where(a > 0, jnp.where(jnp.isfinite(f), f, 0), -jnp.inf)
                for f, a in zip(potentials, marginals, strict=True)
            )
        else:
            for f, a in zip(potentials, marginals, strict=True):
                valid = valid & jnp.where(a > 0, jnp.isfinite(f), jnp.isneginf(f)).all()
    count = (
        jnp.asarray(0, jnp.int32) if state is None else jnp.asarray(state.iterations)
    )
    valid = valid & jnp.isfinite(count) & (count >= 0) & (count == jnp.floor(count))
    count = count.astype(jnp.int32)
    stop = count + budget.astype(jnp.int32)
    valid = valid & (stop >= count)
    error = _error(cost, marginals, potentials, epsilon)

    def condition(carry):
        iteration, error, _, failed = carry
        return valid & ~failed & (iteration < stop) & ~(error <= tolerance)

    def body(carry):
        iteration, _, current, _ = carry
        updated = list(current)
        for axis, marginal in enumerate(marginals):
            value = -cost
            for other, potential in enumerate(updated):
                if other != axis:
                    value = value + _broadcast(potential, other, cost.ndim)
            axes = tuple(i for i in range(cost.ndim) if i != axis)
            reduction = logsumexp(value / epsilon, axis=axes)
            updated[axis] = jnp.where(
                marginal > 0, epsilon * (jnp.log(marginal) - reduction), -jnp.inf
            )
        normalized = _gauge(tuple(updated), marginals)
        error = _error(cost, marginals, normalized, epsilon)
        return iteration + 1, error, normalized, ~jnp.isfinite(error)

    count, error, potentials, failed = jax.lax.while_loop(
        condition, body, (count, error, potentials, jnp.asarray(False))
    )
    status = jnp.where(
        ~valid,
        INVALID_INPUT,
        jnp.where(
            failed | ~jnp.isfinite(error),
            NUMERICAL_FAILURE,
            jnp.where(error <= tolerance, CONVERGED, ITERATION_LIMIT),
        ),
    )
    return SinkhornState(potentials, count), SolverDiagnostics(error, count, status)
