# SPDX-License-Identifier: Apache-2.0
"""Common floating precision for flow inputs and loop carry arrays."""

import jax
import jax.numpy as jnp


def flow_dtype(cost, rho, energy):
    arrays = [
        value
        for value in jax.tree.leaves(energy)
        if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating)
    ]
    return jnp.result_type(cost, rho, *arrays)


def cast_floating(tree, dtype):
    def cast(value):
        array = jnp.asarray(value)
        return (
            array.astype(dtype) if jnp.issubdtype(array.dtype, jnp.floating) else value
        )

    return jax.tree.map(cast, tree)
