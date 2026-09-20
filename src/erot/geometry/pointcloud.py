"""Squared Euclidean cost tiles without a dense cost matrix."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .dense import block_indices


class PointCloudGeometry(NamedTuple):
    """Two point arrays (count, features); only tile differences are formed."""

    x: jax.Array
    y: jax.Array

    @property
    def shape(self) -> tuple[int, int]:
        if self.x.ndim != 2 or self.y.ndim != 2:
            raise ValueError("point clouds must have shape (count, features)")
        if min(*self.x.shape, *self.y.shape) == 0 or self.x.shape[1] != self.y.shape[1]:
            raise ValueError("point clouds need positive counts and matching features")
        if any(not jnp.issubdtype(a.dtype, jnp.floating) for a in (self.x, self.y)):
            raise ValueError("point coordinates must have real floating dtypes")
        return self.x.shape[0], self.y.shape[0]

    @property
    def dtype(self):
        return jnp.result_type(self.x, self.y)

    def cost_block(self, row_start, column_start, block_size):
        n, m = self.shape
        rows, valid_rows = block_indices(row_start, block_size, n)
        columns, valid_columns = block_indices(column_start, block_size, m)
        # Direct differences avoid cancellation from ||x||²+||y||²-2<x,y>.
        difference = self.x[rows, None, :] - self.y[None, columns, :]
        tile = jnp.sum(difference * difference, axis=-1)
        return jnp.where(valid_rows[:, None] & valid_columns[None, :], tile, jnp.inf)

    def transpose(self):
        return PointCloudGeometry(self.y, self.x)

    def is_valid(self):
        return jnp.isfinite(self.x).all() & jnp.isfinite(self.y).all()
