# SPDX-License-Identifier: Apache-2.0
"""Discrete nonnegative cell-mass energies with value, gradient and proximal maps."""

from __future__ import annotations

from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
from jax.scipy.special import xlogy

from ..optim.prox import entropy_prox


class Energy(Protocol):
    """Device-compatible energy interface; concrete energies are PyTrees."""

    def value(self, rho: jax.Array) -> jax.Array: ...
    def gradient(self, rho: jax.Array) -> jax.Array: ...
    def prox(self, z: jax.Array, alpha: jax.Array) -> jax.Array: ...
    def is_valid(self) -> jax.Array: ...


class Entropy(NamedTuple):
    """weight * sum rho*(log(rho/volumes)-1), with positive cell volumes."""

    volumes: jax.Array
    weight: float = 1.0

    def is_valid(self):
        w = jnp.asarray(self.volumes)
        return (
            jnp.isfinite(w).all()
            & (w > 0).all()
            & jnp.isfinite(self.weight)
            & (self.weight >= 0)
        )

    def value(self, rho):
        active_rho = jnp.where(self.weight == 0, jnp.ones_like(rho), rho)
        value = self.weight * jnp.sum(
            xlogy(rho, active_rho) - rho * jnp.log(self.volumes) - rho
        )
        return jnp.where(self.is_valid() & (rho >= 0).all(), value, jnp.inf)

    def gradient(self, rho):
        active_rho = jnp.where(self.weight == 0, jnp.ones_like(rho), rho)
        return self.weight * (jnp.log(active_rho) - jnp.log(self.volumes))

    def prox(self, z, alpha):
        return entropy_prox(z, alpha * self.weight, self.volumes)


class Quadratic(NamedTuple):
    """weight/2 * squared Euclidean distance to target, on nonnegative masses."""

    target: jax.Array
    weight: float = 1.0

    def is_valid(self):
        return (
            jnp.isfinite(self.target).all()
            & jnp.isfinite(self.weight)
            & (self.weight >= 0)
        )

    def value(self, rho):
        value = self.weight / 2 * jnp.sum((rho - self.target) ** 2)
        return jnp.where(self.is_valid() & (rho >= 0).all(), value, jnp.inf)

    def gradient(self, rho):
        return self.weight * (rho - self.target)

    def prox(self, z, alpha):
        return jnp.maximum(
            (z + alpha * self.weight * self.target) / (1 + alpha * self.weight), 0.0
        )


class Potential(NamedTuple):
    """sum potential*rho on nonnegative masses."""

    potential: jax.Array

    def is_valid(self):
        return jnp.isfinite(self.potential).all()

    def value(self, rho):
        return jnp.where(
            self.is_valid() & (rho >= 0).all(), jnp.sum(self.potential * rho), jnp.inf
        )

    def gradient(self, rho):
        return jnp.broadcast_to(self.potential, rho.shape)

    def prox(self, z, alpha):
        return jnp.maximum(z - alpha * self.potential, 0.0)
