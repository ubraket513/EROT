"""Stable log reductions with tile-bounded cost and coupling intermediates."""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from .dense import block_indices


def streamed_logsumexp(geometry, left, right, epsilon, *, block_size=128):
    """Return row log-sums of exp((left_i+right_j-C_ij)/epsilon).

    Both vectors contain cost-unit potentials; inactive entries are -inf.
    Use zero left potentials to compute the reduction for a Sinkhorn update.
    Invalid padding contributes -inf, even for completely inactive tiles.
    """
    n, m = geometry.shape
    if left.shape != (n,) or right.shape != (m,):
        raise ValueError("potential shapes must match geometry")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive static integer")
    row_blocks = (n + block_size - 1) // block_size
    column_blocks = (m + block_size - 1) // block_size
    dtype = jnp.result_type(geometry.dtype, left, right)
    output = jnp.full((row_blocks * block_size,), -jnp.inf, dtype)

    def row_body(row, output):
        row_start = row * block_size
        rows, row_valid = block_indices(row_start, block_size, n)
        left_tile = jnp.where(row_valid, left[rows], -jnp.inf)

        def column_body(column, accumulated):
            column_start = column * block_size
            columns, column_valid = block_indices(column_start, block_size, m)
            right_tile = jnp.where(column_valid, right[columns], -jnp.inf)
            cost = geometry.cost_block(row_start, column_start, block_size)
            exponent = (left_tile[:, None] + right_tile[None, :] - cost) / epsilon
            return jnp.logaddexp(accumulated, logsumexp(exponent, axis=1))

        reduced = jax.lax.fori_loop(
            0, column_blocks, column_body, jnp.full((block_size,), -jnp.inf, dtype)
        )
        return jax.lax.dynamic_update_slice(output, reduced, (row_start,))

    return jax.lax.fori_loop(0, row_blocks, row_body, output)[:n]
