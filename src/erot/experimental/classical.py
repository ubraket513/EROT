"""Experimental quadratic dual solvers.

These kernels have corrected state flow and compiled loops, but remain outside
the stable API until their convergence properties are independently validated.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..classical import _classical_error


def _coupling(
    cost: jax.Array, f: jax.Array, g: jax.Array, epsilon: jax.Array
) -> jax.Array:
    return jnp.maximum(f[:, None] + g[None, :] - cost, 0) / epsilon


@jax.jit
def quadratic_gradient_descent(
    cost: jax.Array,
    marginal_a: jax.Array,
    marginal_b: jax.Array,
    epsilon: jax.Array,
    tolerance: jax.Array,
    max_iterations: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    step = epsilon / (marginal_a.shape[0] + marginal_b.shape[0])
    initial = (
        jnp.asarray(0, jnp.int32),
        jnp.asarray(jnp.inf, cost.dtype),
        jnp.zeros_like(marginal_a),
        jnp.zeros_like(marginal_b),
    )

    def condition(state: tuple[jax.Array, ...]) -> jax.Array:
        iteration, error, *_ = state
        return (iteration < max_iterations) & (error > tolerance)

    def body(state: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        iteration, _, f, g = state
        coupling = _coupling(cost, f, g, epsilon)
        new_f = f + step * (marginal_a - coupling.sum(axis=1))
        new_g = g + step * (marginal_b - coupling.sum(axis=0))
        new_coupling = _coupling(cost, new_f, new_g, epsilon)
        error = _classical_error(new_coupling, (marginal_a, marginal_b))
        return iteration + 1, error, new_f, new_g

    iterations, error, f, g = jax.lax.while_loop(condition, body, initial)
    return _coupling(cost, f, g, epsilon), error, iterations


@jax.jit
def quadratic_fixed_point(
    cost: jax.Array,
    marginal_a: jax.Array,
    marginal_b: jax.Array,
    epsilon: jax.Array,
    tolerance: jax.Array,
    max_iterations: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    n, m = marginal_a.shape[0], marginal_b.shape[0]
    initial = (
        jnp.asarray(0, jnp.int32),
        jnp.asarray(jnp.inf, cost.dtype),
        jnp.zeros_like(marginal_a),
        jnp.zeros_like(marginal_b),
    )

    def condition(state: tuple[jax.Array, ...]) -> jax.Array:
        iteration, error, *_ = state
        return (iteration < max_iterations) & (error > tolerance)

    def body(state: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        iteration, _, f, g = state
        coupling = _coupling(cost, f, g, epsilon)
        update_f = -epsilon * (coupling.sum(axis=1) - marginal_a)
        new_f = f + (update_f - update_f.mean()) / m
        coupling_after_f = _coupling(cost, new_f, g, epsilon)
        update_g = -epsilon * (coupling_after_f.sum(axis=0) - marginal_b)
        new_g = g + (update_g - update_g.mean()) / n
        new_coupling = _coupling(cost, new_f, new_g, epsilon)
        error = _classical_error(new_coupling, (marginal_a, marginal_b))
        return iteration + 1, error, new_f, new_g

    iterations, error, f, g = jax.lax.while_loop(condition, body, initial)
    return _coupling(cost, f, g, epsilon), error, iterations


@jax.jit
def quadratic_nesterov(
    cost: jax.Array,
    marginal_a: jax.Array,
    marginal_b: jax.Array,
    epsilon: jax.Array,
    tolerance: jax.Array,
    max_iterations: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    step = epsilon / (marginal_a.shape[0] + marginal_b.shape[0])
    zeros_a, zeros_b = jnp.zeros_like(marginal_a), jnp.zeros_like(marginal_b)
    initial = (
        jnp.asarray(0, jnp.int32),
        jnp.asarray(jnp.inf, cost.dtype),
        zeros_a,
        zeros_b,
        zeros_a,
        zeros_b,
    )

    def condition(state: tuple[jax.Array, ...]) -> jax.Array:
        iteration, error, *_ = state
        return (iteration < max_iterations) & (error > tolerance)

    def body(state: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        iteration, _, f, g, previous_f, previous_g = state
        momentum = jnp.maximum(iteration - 1, 0) / (iteration + 2)
        extrapolated_f = f + momentum * (f - previous_f)
        extrapolated_g = g + momentum * (g - previous_g)
        coupling = _coupling(cost, extrapolated_f, extrapolated_g, epsilon)
        new_f = extrapolated_f + step * (marginal_a - coupling.sum(axis=1))
        new_g = extrapolated_g + step * (marginal_b - coupling.sum(axis=0))
        new_coupling = _coupling(cost, new_f, new_g, epsilon)
        error = _classical_error(new_coupling, (marginal_a, marginal_b))
        return iteration + 1, error, new_f, new_g, f, g

    iterations, error, f, g, *_ = jax.lax.while_loop(condition, body, initial)
    return _coupling(cost, f, g, epsilon), error, iterations
