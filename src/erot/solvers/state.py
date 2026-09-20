"""Array-only solver state and status codes suitable for JAX transformations."""

from typing import NamedTuple

import jax

CONVERGED = 0
ITERATION_LIMIT = 1
INVALID_INPUT = 2
NUMERICAL_FAILURE = 3


class SinkhornState(NamedTuple):
    """Exact resume data for an unchanged problem; potentials are in cost units."""

    potentials: tuple[jax.Array, ...]
    iterations: jax.Array


class SinkhornWarmStart(NamedTuple):
    """Potentials for a related problem; work counts and convergence are reset."""

    potentials: tuple[jax.Array, ...]


class SolverDiagnostics(NamedTuple):
    """Returned-iterate marginal L1 error, cumulative sweeps and status code."""

    error: jax.Array
    iterations: jax.Array
    status: jax.Array
