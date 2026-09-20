"""Explicit transport tiles, products and objectives without a full coupling."""

import jax
import jax.numpy as jnp
from jax.scipy.special import xlogy

from .dense import block_indices


def plan_block(
    geometry, potentials, epsilon, row_start, column_start, *, block_size=128
):
    """Return a fixed-size coupling tile, with padding entries exactly zero."""
    n, m = geometry.shape
    f, g = potentials
    if f.shape != (n,) or g.shape != (m,):
        raise ValueError("potential shapes must match geometry")
    rows, valid_rows = block_indices(row_start, block_size, n)
    columns, valid_columns = block_indices(column_start, block_size, m)
    cost = geometry.cost_block(row_start, column_start, block_size)
    value = jnp.exp((f[rows, None] + g[None, columns] - cost) / epsilon)
    return jnp.where(valid_rows[:, None] & valid_columns[None, :], value, 0)


def transport_objective(geometry, potentials, epsilon, *, block_size=128):
    """Stream sum C*pi + epsilon*pi*(log(pi)-1), with 0 log 0 = 0."""
    n, m = geometry.shape
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive static integer")
    nr, nc = (n + block_size - 1) // block_size, (m + block_size - 1) // block_size

    def add_tile(index, total):
        row, column = (index // nc) * block_size, (index % nc) * block_size
        coupling = plan_block(
            geometry, potentials, epsilon, row, column, block_size=block_size
        )
        cost = geometry.cost_block(row, column, block_size)
        _, vr = block_indices(row, block_size, n)
        _, vc = block_indices(column, block_size, m)
        cost = jnp.where(vr[:, None] & vc[None, :], cost, 0)
        return total + jnp.sum(
            cost * coupling + epsilon * (xlogy(coupling, coupling) - coupling)
        )

    dtype = jnp.result_type(geometry.dtype, *potentials, epsilon)
    return jax.lax.fori_loop(0, nr * nc, add_tile, jnp.asarray(0, dtype))


def apply_transport(geometry, potentials, epsilon, values, *, block_size=128):
    """Compute pi @ values for a vector or feature matrix using bounded tiles.

    For pi.T @ values, transpose geometry and reverse the potential tuple.
    """
    n, m = geometry.shape
    values = jnp.asarray(values)
    vector = values.ndim == 1
    if values.ndim not in (1, 2) or values.shape[0] != m:
        raise ValueError("values must have shape (columns,) or (columns, features)")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive static integer")
    if vector:
        values = values[:, None]
    nr, nc = (n + block_size - 1) // block_size, (m + block_size - 1) // block_size
    dtype = jnp.result_type(geometry.dtype, *potentials, values, epsilon)
    output = jnp.zeros((nr * block_size, values.shape[1]), dtype)

    def add_row(row, output):
        def add_column(column, partial):
            indices, valid = block_indices(column * block_size, block_size, m)
            tile = plan_block(
                geometry,
                potentials,
                epsilon,
                row * block_size,
                column * block_size,
                block_size=block_size,
            )
            features = jnp.where(valid[:, None], values[indices], 0)
            return partial + tile @ features

        partial = jax.lax.fori_loop(
            0, nc, add_column, jnp.zeros((block_size, values.shape[1]), dtype)
        )
        return jax.lax.dynamic_update_slice(output, partial, (row * block_size, 0))

    output = jax.lax.fori_loop(0, nr, add_row, output)[:n]
    return output[:, 0] if vector else output
