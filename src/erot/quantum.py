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

    from .solvers.quantum import solve_quantum_quadratic

    state, diagnostics = solve_quantum_quadratic(
        cost, marginal_a, marginal_b, epsilon, tolerance, max_iterations
    )
    return state.coupling, diagnostics.error, diagnostics.iterations
