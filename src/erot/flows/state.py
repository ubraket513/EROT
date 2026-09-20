# SPDX-License-Identifier: Apache-2.0
"""Array-only state and diagnostics for transport gradient-flow subproblems."""

from typing import NamedTuple

import jax


class PDHGState(NamedTuple):
    """Complete resume state for an unchanged unregularized JKO subproblem."""

    coupling: jax.Array
    rho: jax.Array
    dual_rows: jax.Array
    dual_columns: jax.Array
    extrapolated_coupling: jax.Array
    extrapolated_rho: jax.Array
    iterations: jax.Array


class JKODiagnostics(NamedTuple):
    """Objective, feasibility and fixed-point stationarity at the returned iterate."""

    objective: jax.Array
    feasibility: jax.Array
    stationarity: jax.Array
    error: jax.Array
    iterations: jax.Array
    status: jax.Array


INNER_SOLVE_FAILED = 4
LINE_SEARCH_FAILED = 5


class EntropicJKOState(NamedTuple):
    """Outer density, reusable transport potentials and cumulative work counts."""

    rho: jax.Array
    potentials: tuple[jax.Array, ...]
    iterations: jax.Array
    inner_iterations: jax.Array
    step_size: jax.Array
