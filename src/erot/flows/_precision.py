# SPDX-License-Identifier: Apache-2.0
"""Common floating precision for flow inputs and loop carry arrays."""

import jax
import jax.numpy as jnp

from ..geometry import DenseGeometry, PointCloudGeometry


def is_geometry(cost):
    return isinstance(cost, (DenseGeometry, PointCloudGeometry))


def as_flow_cost(cost):
    return cost if is_geometry(cost) else jnp.asarray(cost)


def flow_dtype(cost, rho, energy):
    arrays = [
        value
        for value in jax.tree.leaves(energy)
        if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating)
    ]
    return jnp.result_type(cost.dtype, rho, *arrays)


def cast_floating(tree, dtype):
    def cast(value):
        array = jnp.asarray(value)
        return (
            array.astype(dtype) if jnp.issubdtype(array.dtype, jnp.floating) else value
        )

    return jax.tree.map(cast, tree)
