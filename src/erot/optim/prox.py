"""Stable elementwise proximal maps used by discrete gradient flows."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def entropy_prox(z: jax.Array, alpha: jax.Array, volumes: jax.Array = 1.0) -> jax.Array:
    """Prox of alpha * sum x*(log(x/volume)-1) on x>=0.

    Solve x + alpha*log(x/volume) = z in log coordinates. Bisection never
    exponentiates z/alpha. Roots below the dtype's normal range return zero,
    consistent with accelerator underflow. alpha=0 projects onto x>=0.
    Invalid alpha/volumes produce NaNs for the caller to diagnose.
    """
    z = jnp.asarray(z)
    alpha, volumes = jnp.asarray(alpha, z.dtype), jnp.asarray(volumes, z.dtype)
    safe_alpha = jnp.where(alpha > 0, alpha, 1.0)
    log_volume = jnp.log(volumes)
    lower = jnp.full_like(z, jnp.log(jnp.finfo(z.dtype).tiny))
    upper = jnp.maximum(jnp.log(jnp.maximum(z, 1.0)), log_volume) + 1.0

    def equation(log_x):
        return jnp.exp(log_x) + safe_alpha * (log_x - log_volume) - z

    def bisect(_, bounds):
        lo, hi = bounds
        middle = lo + (hi - lo) / 2
        positive = equation(middle) > 0
        return jnp.where(positive, lo, middle), jnp.where(positive, middle, hi)

    lo, hi = jax.lax.fori_loop(0, 80, bisect, (lower, upper))
    result = jnp.where(equation(lower) >= 0, 0.0, jnp.exp((lo + hi) / 2))
    result = jnp.where(alpha == 0, jnp.maximum(z, 0.0), result)
    valid = (
        jnp.isfinite(z)
        & jnp.isfinite(alpha)
        & (alpha >= 0)
        & jnp.isfinite(volumes)
        & (volumes > 0)
    )
    return jnp.where(valid, result, jnp.nan)
