"""Compiled quantum entropy-regularized optimal transport solvers."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _hermitian(matrix: jax.Array) -> jax.Array:
    return (matrix + matrix.conj().T) / 2


def partial_trace_second(operator: jax.Array, n: int, m: int) -> jax.Array:
    """Trace out the second, ``m``-dimensional Hilbert space."""

    blocks = operator.reshape(n, m, n, m)
    return jnp.einsum("iaja->ij", blocks)


def partial_trace_first(operator: jax.Array, n: int, m: int) -> jax.Array:
    """Trace out the first, ``n``-dimensional Hilbert space."""

    blocks = operator.reshape(n, m, n, m)
    return jnp.einsum("iaib->ab", blocks)


def _lift_first(matrix: jax.Array, identity_m: jax.Array) -> jax.Array:
    n, m = matrix.shape[0], identity_m.shape[0]
    return jnp.einsum("ij,ab->iajb", matrix, identity_m).reshape(n * m, n * m)


def _lift_second(identity_n: jax.Array, matrix: jax.Array) -> jax.Array:
    n, m = identity_n.shape[0], matrix.shape[0]
    return jnp.einsum("ij,ab->iajb", identity_n, matrix).reshape(n * m, n * m)


def _project_psd(matrix: jax.Array) -> jax.Array:
    eigenvalues, eigenvectors = jnp.linalg.eigh(_hermitian(matrix))
    positive = jnp.maximum(eigenvalues, 0)
    return _hermitian((eigenvectors * positive[None, :]) @ eigenvectors.conj().T)


def _marginal_error(
    coupling: jax.Array, marginal_a: jax.Array, marginal_b: jax.Array
) -> jax.Array:
    n, m = marginal_a.shape[0], marginal_b.shape[0]
    error_a = jnp.sum(jnp.abs(partial_trace_second(coupling, n, m) - marginal_a))
    error_b = jnp.sum(jnp.abs(partial_trace_first(coupling, n, m) - marginal_b))
    return jnp.maximum(error_a, error_b)


@jax.jit
def quadratic_cyclic_projection(
    cost: jax.Array,
    marginal_a: jax.Array,
    marginal_b: jax.Array,
    epsilon: jax.Array,
    tolerance: jax.Array,
    max_iterations: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Project ``-cost / epsilon`` onto the quantum coupling set with Dykstra."""

    n, m = marginal_a.shape[0], marginal_b.shape[0]
    identity_n = jnp.eye(n, dtype=cost.dtype)
    identity_m = jnp.eye(m, dtype=cost.dtype)
    initial = _hermitian(-cost / epsilon)
    correction_a = jnp.zeros_like(cost)
    correction_b = jnp.zeros_like(cost)
    correction_psd = jnp.zeros_like(cost)
    initial_state = (
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(jnp.inf, dtype=cost.real.dtype),
        initial,
        correction_a,
        correction_b,
        correction_psd,
    )

    def condition(state: tuple[jax.Array, ...]) -> jax.Array:
        iteration, error, *_ = state
        return jnp.logical_and(iteration < max_iterations, error > tolerance)

    def body(state: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        iteration, _, coupling, corr_a, corr_b, corr_psd = state

        candidate_a = _hermitian(coupling + corr_a)
        residual_a = marginal_a - partial_trace_second(candidate_a, n, m)
        projected_a = _hermitian(candidate_a + _lift_first(residual_a, identity_m) / m)
        new_corr_a = candidate_a - projected_a

        candidate_b = _hermitian(projected_a + corr_b)
        residual_b = marginal_b - partial_trace_first(candidate_b, n, m)
        projected_b = _hermitian(candidate_b + _lift_second(identity_n, residual_b) / n)
        new_corr_b = candidate_b - projected_b

        candidate_psd = _hermitian(projected_b + corr_psd)
        projected_psd = _project_psd(candidate_psd)
        new_corr_psd = candidate_psd - projected_psd

        error = _marginal_error(projected_psd, marginal_a, marginal_b)
        return (
            iteration + 1,
            error,
            projected_psd,
            new_corr_a,
            new_corr_b,
            new_corr_psd,
        )

    iterations, error, coupling, *_ = jax.lax.while_loop(condition, body, initial_state)
    return coupling, error, iterations
