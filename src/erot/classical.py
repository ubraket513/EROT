"""Compiled classical entropy-regularized optimal transport solvers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def _broadcast_vector(vector: jax.Array, axis: int, rank: int) -> jax.Array:
    shape = [1] * rank
    shape[axis] = vector.shape[0]
    return vector.reshape(shape)


def _log_coupling(
    cost: jax.Array, log_scalings: tuple[jax.Array, ...], epsilon: jax.Array
) -> jax.Array:
    result = -cost / epsilon
    for axis, scaling in enumerate(log_scalings):
        result = result + _broadcast_vector(scaling, axis, cost.ndim)
    return result


def _classical_error(
    coupling: jax.Array, marginals: tuple[jax.Array, ...]
) -> jax.Array:
    error = jnp.asarray(0.0, dtype=coupling.real.dtype)
    all_axes = tuple(range(coupling.ndim))
    for axis, marginal in enumerate(marginals):
        reduction_axes = tuple(index for index in all_axes if index != axis)
        residual = jnp.sum(coupling, axis=reduction_axes) - marginal
        error = jnp.maximum(error, jnp.sum(jnp.abs(residual)))
    return error


@jax.jit
def shannon_sinkhorn(
    cost: jax.Array,
    marginals: tuple[jax.Array, ...],
    epsilon: jax.Array,
    tolerance: jax.Array,
    max_iterations: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Solve dense multi-marginal Shannon EROT with log-domain Sinkhorn updates."""

    dtype = cost.dtype
    log_scalings = tuple(jnp.zeros_like(marginal) for marginal in marginals)
    initial_state = (
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(jnp.inf, dtype=dtype),
        log_scalings,
    )
    all_axes = tuple(range(cost.ndim))

    def condition(
        state: tuple[jax.Array, jax.Array, tuple[jax.Array, ...]],
    ) -> jax.Array:
        iteration, error, _ = state
        return jnp.logical_and(iteration < max_iterations, error > tolerance)

    def body(
        state: tuple[jax.Array, jax.Array, tuple[jax.Array, ...]],
    ) -> tuple[jax.Array, jax.Array, tuple[jax.Array, ...]]:
        iteration, _, current_scalings = state
        updated_scalings = list(current_scalings)
        for axis, marginal in enumerate(marginals):
            log_values = _log_coupling(cost, tuple(updated_scalings), epsilon)
            reduction_axes = tuple(index for index in all_axes if index != axis)
            log_marginal = logsumexp(log_values, axis=reduction_axes)
            log_target = jnp.where(marginal > 0, jnp.log(marginal), -jnp.inf)
            updated_scalings[axis] = jnp.where(
                marginal > 0,
                updated_scalings[axis] + log_target - log_marginal,
                -jnp.inf,
            )

        coupling = jnp.exp(_log_coupling(cost, tuple(updated_scalings), epsilon))
        error = _classical_error(coupling, marginals)
        return iteration + 1, error, tuple(updated_scalings)

    iterations, error, final_scalings = jax.lax.while_loop(
        condition, body, initial_state
    )
    coupling = jnp.exp(_log_coupling(cost, final_scalings, epsilon))
    return coupling, error, iterations


def _positive_part_thresholds(z: jax.Array, targets: jax.Array) -> jax.Array:
    """Find t row-wise such that ``sum(max(t - z, 0)) == targets``."""

    sorted_z = jnp.sort(z, axis=1)
    cumulative = jnp.cumsum(sorted_z, axis=1)
    counts = jnp.arange(1, z.shape[1] + 1, dtype=z.real.dtype)
    candidates = (targets[:, None] + cumulative) / counts[None, :]
    active_counts = jnp.sum(candidates >= sorted_z, axis=1)
    indices = jnp.maximum(active_counts - 1, 0)[:, None]
    return jnp.take_along_axis(candidates, indices, axis=1)[:, 0]


@jax.jit
def quadratic_cyclic_projection(
    cost: jax.Array,
    marginal_a: jax.Array,
    marginal_b: jax.Array,
    epsilon: jax.Array,
    tolerance: jax.Array,
    max_iterations: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Solve two-marginal quadratic EROT by exact cyclic dual projections."""

    f = jnp.zeros_like(marginal_a)
    g = jnp.zeros_like(marginal_b)
    initial_state = (
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(jnp.inf, dtype=cost.dtype),
        f,
        g,
    )

    def condition(state: tuple[jax.Array, ...]) -> jax.Array:
        iteration, error, _, _ = state
        return jnp.logical_and(iteration < max_iterations, error > tolerance)

    def body(state: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        iteration, _, _, current_g = state
        new_f = _positive_part_thresholds(
            cost - current_g[None, :], epsilon * marginal_a
        )
        new_g = _positive_part_thresholds(
            (cost - new_f[:, None]).T, epsilon * marginal_b
        )
        coupling = jnp.maximum(new_f[:, None] + new_g[None, :] - cost, 0) / epsilon
        error = _classical_error(coupling, (marginal_a, marginal_b))
        return iteration + 1, error, new_f, new_g

    iterations, error, final_f, final_g = jax.lax.while_loop(
        condition, body, initial_state
    )
    coupling = jnp.maximum(final_f[:, None] + final_g[None, :] - cost, 0) / epsilon
    return coupling, error, iterations
