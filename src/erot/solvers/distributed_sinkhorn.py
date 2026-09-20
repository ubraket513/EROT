"""Experimental row-partitioned implicit Sinkhorn with explicit collectives."""

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from ..geometry import PointCloudGeometry
from ..geometry.reductions import streamed_logsumexp
from ._controls import _checked_count
from .state import (
    CONVERGED,
    INVALID_INPUT,
    ITERATION_LIMIT,
    NUMERICAL_FAILURE,
    SinkhornState,
    SolverDiagnostics,
)


def _global_logsumexp(local):
    """Combine per-rank log-sums, including an entirely empty support."""
    maximum = jax.lax.pmax(local, "rows")
    safe = jnp.where(jnp.isneginf(maximum), 0, maximum)
    summed = jax.lax.psum(jnp.exp(local - safe), "rows")
    return safe + jnp.log(summed)


def make_distributed_sinkhorn(mesh, block_size=128):
    """Return a compiled solver for global, row-sharded squared-Euclidean OT.

    Signature: solve(x, y, a, b, epsilon, tolerance, max_iterations, state).
    x/a/f are row-sharded; y/b/g, scalar controls and diagnostics are replicated.
    Source padding must have zero mass and finite coordinates. All input ranks
    must supply identical replicated data. State resumes the identical problem;
    max_iterations is additional work. This experimental API uses shard_map.
    """
    if mesh.axis_names != ("rows",):
        raise ValueError("expected a one-dimensional rows mesh")
    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be a positive static integer")
    state_spec = SinkhornState((P("rows"), P()), P())

    def local_solve(x, y, a, b, epsilon, tolerance, max_iterations, state):
        geometry = PointCloudGeometry(x, y)
        n, m = geometry.shape
        if a.shape != (n,) or b.shape != (m,):
            raise ValueError("marginal lengths must match geometry")
        if len(state.potentials) != 2 or tuple(f.shape for f in state.potentials) != (
            (n,),
            (m,),
        ):
            raise ValueError("state potential shapes must match marginals")
        if any(
            jnp.iscomplexobj(value)
            for value in (a, b, epsilon, tolerance, *state.potentials)
        ):
            raise ValueError("classical inputs must be real")
        if any(
            jnp.ndim(value) != 0
            for value in (epsilon, tolerance, max_iterations, state.iterations)
        ):
            raise ValueError("numerical controls and iteration counts must be scalars")
        dtype = geometry.dtype
        a, b = a.astype(dtype), b.astype(dtype)
        f, g = (value.astype(dtype) for value in state.potentials)
        eps, tol = (jnp.asarray(value, dtype) for value in (epsilon, tolerance))
        budget, valid_budget = _checked_count(jnp.asarray(max_iterations))
        count, valid_count = _checked_count(jnp.asarray(state.iterations))
        fits = budget <= 2**31 - 1 - count
        stop = count + jnp.where(fits, budget, 0)
        mass = jax.lax.psum(a.sum(), "rows")
        mass_tolerance = 1e-6 if dtype == jnp.float32 else 1e-10
        valid = geometry.is_valid() & valid_budget & valid_count & fits
        valid &= jnp.isfinite(eps) & (eps > 0) & jnp.isfinite(tol) & (tol > 0)
        valid &= jnp.isfinite(a).all() & (a >= 0).all()
        valid &= jnp.isfinite(b).all() & (b >= 0).all() & (mass > 0)
        valid &= jnp.isclose(mass, b.sum(), rtol=mass_tolerance, atol=mass_tolerance)
        valid &= jnp.where(a > 0, jnp.isfinite(f), jnp.isneginf(f)).all()
        valid &= jnp.where(b > 0, jnp.isfinite(g), jnp.isneginf(g)).all()
        valid = jax.lax.pmin(valid.astype(jnp.int32), "rows").astype(jnp.bool_)

        def rows(f, g):
            return streamed_logsumexp(
                geometry, f, g, eps, block_size=block_size, varying_axis="rows"
            )

        def columns(f, g):
            local = streamed_logsumexp(
                geometry.transpose(),
                g,
                f,
                eps,
                block_size=block_size,
                varying_axis="rows",
            )
            return _global_logsumexp(local)

        def residual(f, g):
            row_error = jax.lax.psum(jnp.abs(jnp.exp(rows(f, g)) - a).sum(), "rows")
            column_error = jnp.abs(jnp.exp(columns(f, g)) - b).sum()
            return jnp.maximum(row_error, column_error)

        def condition(carry):
            iteration, error, _, _, failed = carry
            return valid & ~failed & (iteration < stop) & ~(error <= tol)

        def update(carry):
            iteration, _, f, g, _ = carry
            f = jnp.where(
                a > 0, eps * (jnp.log(a) - rows(jnp.zeros_like(f), g)), -jnp.inf
            )
            g = jnp.where(
                b > 0, eps * (jnp.log(b) - columns(f, jnp.zeros_like(g))), -jnp.inf
            )
            weighted = jnp.where(a > 0, f, 0) * a
            shift = jax.lax.psum(weighted.sum(), "rows") / mass
            f = jnp.where(a > 0, f - shift, -jnp.inf)
            g = jnp.where(b > 0, g + shift, -jnp.inf)
            error = residual(f, g)
            return iteration + 1, error, f, g, ~jnp.isfinite(error)

        error = residual(f, g)
        count, error, f, g, failed = jax.lax.while_loop(
            condition, update, (count, error, f, g, jnp.asarray(False))
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
        return SinkhornState((f, g), count), SolverDiagnostics(error, count, status)

    mapped = jax.shard_map(
        local_solve,
        mesh=mesh,
        in_specs=(P("rows"), P(), P("rows"), P(), P(), P(), P(), state_spec),
        out_specs=(state_spec, SolverDiagnostics(P(), P(), P())),
        check_vma=True,
    )
    return jax.jit(mapped)
