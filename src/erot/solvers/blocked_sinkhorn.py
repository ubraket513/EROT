"""Two-marginal Sinkhorn using bounded geometry tiles and streamed residuals."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..geometry.reductions import streamed_logsumexp
from ._controls import _checked_count
from .sinkhorn import _gauge
from .state import (
    CONVERGED,
    INVALID_INPUT,
    ITERATION_LIMIT,
    NUMERICAL_FAILURE,
    SinkhornState,
    SinkhornWarmStart,
    SolverDiagnostics,
)


def solve_blocked_sinkhorn(
    geometry,
    marginals: tuple[jax.Array, jax.Array],
    epsilon: jax.Array,
    tolerance: jax.Array,
    max_iterations: jax.Array,
    *,
    block_size: int = 128,
    state: SinkhornState | None = None,
    warm_start: SinkhornWarmStart | None = None,
) -> tuple[SinkhornState, SolverDiagnostics]:
    """Solve Shannon OT without forming a full point-cloud cost or plan.

    Block size is static. Resume state requires identical inputs and controls;
    warm_start reuses potentials for changed problems and resets work counts.
    Returned residual is the maximum marginal L1 error, matching dense Sinkhorn.
    """
    shape = geometry.shape
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive static integer")
    if len(marginals) != 2:
        raise ValueError("blocked Sinkhorn requires two marginals")
    marginals = tuple(jnp.asarray(a) for a in marginals)
    if any(jnp.iscomplexobj(a) for a in marginals):
        raise ValueError("classical marginals must be real")
    if any(a.shape != (n,) for a, n in zip(marginals, shape)):
        raise ValueError("marginal lengths must match geometry")
    if state is not None and warm_start is not None:
        raise ValueError("supply state or warm_start, not both")
    dtype = geometry.dtype
    marginals = tuple(a.astype(dtype) for a in marginals)
    eps, tol = (jnp.asarray(x, dtype) for x in (epsilon, tolerance))
    budget, valid_budget = _checked_count(jnp.asarray(max_iterations))
    valid = geometry.is_valid() & valid_budget & jnp.isfinite(eps) & (eps > 0)
    valid = valid & jnp.isfinite(tol) & (tol > 0)
    mass = marginals[0].sum()
    mass_tol = 1e-6 if dtype == jnp.float32 else 1e-10
    for a in marginals:
        valid = valid & jnp.isfinite(a).all() & (a >= 0).all() & (a.sum() > 0)
        valid = valid & jnp.isclose(a.sum(), mass, rtol=mass_tol, atol=mass_tol)
    initial = state if state is not None else warm_start
    if initial is None:
        potentials = tuple(
            jnp.where(a > 0, jnp.zeros_like(a), -jnp.inf) for a in marginals
        )
    else:
        if len(initial.potentials) != 2 or any(
            f.shape != a.shape for f, a in zip(initial.potentials, marginals)
        ):
            raise ValueError("potential shapes must match marginals")
        potentials = tuple(jnp.asarray(f, dtype) for f in initial.potentials)
        if warm_start is not None:
            potentials = tuple(
                jnp.where(a > 0, jnp.where(jnp.isfinite(f), f, 0), -jnp.inf)
                for f, a in zip(potentials, marginals)
            )
        else:
            for f, a in zip(potentials, marginals):
                valid = valid & jnp.where(a > 0, jnp.isfinite(f), jnp.isneginf(f)).all()
    count, valid_count = _checked_count(
        jnp.asarray(0, jnp.int32) if state is None else jnp.asarray(state.iterations)
    )
    fits = budget <= 2**31 - 1 - count
    valid = valid & valid_count & fits
    stop = count + jnp.where(fits, budget, 0)
    transpose = geometry.transpose()

    def reduce(geom, f, g):
        return streamed_logsumexp(geom, f, g, eps, block_size=block_size)

    def residual(potentials):
        f, g = potentials
        rows = jnp.exp(reduce(geometry, f, g))
        columns = jnp.exp(reduce(transpose, g, f))
        return jnp.maximum(
            jnp.sum(jnp.abs(rows - marginals[0])),
            jnp.sum(jnp.abs(columns - marginals[1])),
        )

    error = residual(potentials)

    def condition(carry):
        iteration, error, _, failed = carry
        return valid & ~failed & (iteration < stop) & ~(error <= tol)

    def update(carry):
        iteration, _, (f, g), _ = carry
        a, b = marginals
        f = jnp.where(
            a > 0, eps * (jnp.log(a) - reduce(geometry, jnp.zeros_like(f), g)), -jnp.inf
        )
        g = jnp.where(
            b > 0,
            eps * (jnp.log(b) - reduce(transpose, jnp.zeros_like(g), f)),
            -jnp.inf,
        )
        potentials = _gauge((f, g), marginals)
        error = residual(potentials)
        return iteration + 1, error, potentials, ~jnp.isfinite(error)

    count, error, potentials, failed = jax.lax.while_loop(
        condition, update, (count, error, potentials, jnp.asarray(False))
    )
    status = jnp.where(
        ~valid,
        INVALID_INPUT,
        jnp.where(
            failed | ~jnp.isfinite(error),
            NUMERICAL_FAILURE,
            jnp.where(error <= tol, CONVERGED, ITERATION_LIMIT),
        ),
    )
    return SinkhornState(potentials, count), SolverDiagnostics(error, count, status)
