# SPDX-License-Identifier: Apache-2.0
"""Bipartite partial traces, their Hilbert–Schmidt adjoint and Gibbs recovery.

The tensor order is (first, second, first-prime, second-prime). Operator
contracts follow QOTLib; unequal dimensions and explicit stable recovery are
EROT adaptations. See NOTICE for provenance.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def partial_traces(
    matrix: jax.Array, dimensions: tuple[int, int]
) -> tuple[jax.Array, jax.Array]:
    """Return both reduced operators; dimensions are static under compilation."""
    n, m = dimensions
    if n <= 0 or m <= 0 or matrix.shape != (n * m, n * m):
        raise ValueError("positive subsystem dimensions must match the square matrix")
    tensor = matrix.reshape(n, m, n, m)
    return jnp.einsum("iaja->ij", tensor), jnp.einsum("iaib->ab", tensor)


def partial_trace_adjoint(duals: tuple[jax.Array, jax.Array]) -> jax.Array:
    """Return U tensor I + I tensor V, with the complex linear adjoint convention."""
    u, v = duals
    if (
        u.ndim != 2
        or v.ndim != 2
        or u.shape[0] != u.shape[1]
        or v.shape[0] != v.shape[1]
        or min(u.shape[0], v.shape[0]) == 0
    ):
        raise ValueError("dual operators must be nonempty square matrices")
    dtype = jnp.result_type(u, v)
    return jnp.kron(u, jnp.eye(v.shape[0], dtype=dtype)) + jnp.kron(
        jnp.eye(u.shape[0], dtype=dtype), v
    )


def gibbs_state(matrix: jax.Array, epsilon: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return exp(H/epsilon)/Tr exp(H/epsilon) and log Tr exp(H/epsilon).

    Requires a finite Hermitian H and positive epsilon. The solver validates
    these domains. This primitive symmetrizes roundoff and shifts eigenvalues
    before dividing by epsilon; it never forms an unscaled matrix exponential.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.shape[0]:
        raise ValueError("Gibbs recovery requires a nonempty square matrix")
    values, vectors = jnp.linalg.eigh((matrix + matrix.conj().T) / 2)
    maximum = jnp.max(values)
    shifted = (values - maximum) / epsilon
    weights = jax.nn.softmax(shifted)
    density = (vectors * weights[None, :]) @ vectors.conj().T
    return (density + density.conj().T) / 2, maximum / epsilon + logsumexp(shifted)
