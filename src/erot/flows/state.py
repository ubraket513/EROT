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
