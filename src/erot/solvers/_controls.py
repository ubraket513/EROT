# SPDX-License-Identifier: Apache-2.0
"""Device scalar validation shared by bounded iterative solvers."""

import jax
import jax.numpy as jnp


def _checked_count(value: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Check bounds before narrowing; float32 rounds int32 max up to 2**31."""
    if jnp.issubdtype(value.dtype, jnp.floating):
        valid = jnp.isfinite(value) & (value == jnp.floor(value)) & (value < 2**31)
    elif jnp.issubdtype(value.dtype, jnp.integer):
        valid = (
            jnp.ones_like(value, dtype=jnp.bool_)
            if jnp.iinfo(value.dtype).max <= 2**31 - 1
            else value <= 2**31 - 1
        )
    else:
        raise ValueError("iteration controls must be real integer-valued scalars")
    valid = valid & (value >= 0)
    return jnp.where(valid, value, 0).astype(jnp.int32), valid
