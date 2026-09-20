"""Dense cost adapter for the shared tiled reduction contract."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


def block_indices(start, block_size, size):
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive static integer")
    indices = start + jnp.arange(block_size)
    return jnp.clip(indices, 0, size - 1), (indices >= 0) & (indices < size)


class DenseGeometry(NamedTuple):
    """Already stored dense cost; this adapter makes no implicit-memory claim."""

    cost: jax.Array

    @property
    def shape(self) -> tuple[int, int]:
        if self.cost.ndim != 2 or min(self.cost.shape) == 0:
            raise ValueError("dense geometry requires a nonempty matrix")
        if not jnp.issubdtype(self.cost.dtype, jnp.floating):
            raise ValueError("dense costs must have a real floating dtype")
        return self.cost.shape

    @property
    def dtype(self):
        return self.cost.dtype

    def cost_block(self, row_start, column_start, block_size):
        n, m = self.shape
        rows, valid_rows = block_indices(row_start, block_size, n)
        columns, valid_columns = block_indices(column_start, block_size, m)
        tile = jnp.asarray(self.cost)[rows[:, None], columns[None, :]]
        return jnp.where(valid_rows[:, None] & valid_columns[None, :], tile, jnp.inf)

    def transpose(self):
        return DenseGeometry(self.cost.T)

    def is_valid(self):
        return jnp.isfinite(self.cost).all()
