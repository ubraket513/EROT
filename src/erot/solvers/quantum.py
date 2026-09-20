"""Dense quadratic quantum OT with complete resumable Dykstra state."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..quantum import (
    _hermitian,
    _lift_first,
    _lift_second,
    _marginal_error,
    _project_psd,
    partial_trace_first,
    partial_trace_second,
)
from .state import (
    CONVERGED,
    INVALID_INPUT,
    ITERATION_LIMIT,
    NUMERICAL_FAILURE,
    SolverDiagnostics,
)


class QuantumDykstraState(NamedTuple):
    """Complete state for identical cost, marginals, epsilon and dtype only."""

    coupling: jax.Array
    correction_a: jax.Array
    correction_b: jax.Array
    correction_psd: jax.Array
    iterations: jax.Array


def solve_quantum_quadratic(
    cost: jax.Array,
    marginal_a: jax.Array,
    marginal_b: jax.Array,
    epsilon: jax.Array,
    tolerance: jax.Array,
    max_iterations: jax.Array,
    *,
    state: QuantumDykstraState | None = None,
) -> tuple[QuantumDykstraState, SolverDiagnostics]:
    """Project -C/epsilon onto PSD matrices with prescribed partial traces.

    Inputs are finite Hermitian arrays, with PSD trace-one marginals. The device
    core checks finiteness, Hermiticity, traces and controls; the host API also
    validates marginal PSD. Budget is additional sweeps. Corrections cannot be
    reused for a changed problem. No host conversions or synchronization occur.
    """
    cost = jnp.asarray(cost)
    marginal_a, marginal_b = jnp.asarray(marginal_a), jnp.asarray(marginal_b)
    n, m = marginal_a.shape[0], marginal_b.shape[0]
    if (
        marginal_a.shape != (n, n)
        or marginal_b.shape != (m, m)
        or cost.shape != (n * m, n * m)
    ):
        raise ValueError("quantum shapes must match the two subsystem dimensions")
    if not jnp.issubdtype(cost.dtype, jnp.inexact):
        raise ValueError("cost must have a floating or complex dtype")
    cost = cost.astype(jnp.result_type(cost, marginal_a, marginal_b))
    marginal_a, marginal_b = (
        marginal_a.astype(cost.dtype),
        marginal_b.astype(cost.dtype),
    )
    epsilon, tolerance = (
        jnp.asarray(epsilon, cost.real.dtype),
        jnp.asarray(tolerance, cost.real.dtype),
    )
    budget = jnp.asarray(max_iterations)
    valid = (
        jnp.isfinite(epsilon)
        & (epsilon > 0)
        & jnp.isfinite(tolerance)
        & (tolerance > 0)
        & jnp.isfinite(budget)
        & (budget >= 0)
        & (budget == jnp.floor(budget))
    )
    input_tolerance = 1e-5 if cost.real.dtype == jnp.float32 else 1e-9
    for matrix in (cost, marginal_a, marginal_b):
        valid = (
            valid
            & jnp.isfinite(matrix).all()
            & (jnp.max(jnp.abs(matrix - matrix.conj().T)) <= input_tolerance)
        )
    for marginal in (marginal_a, marginal_b):
        valid = valid & (jnp.abs(jnp.trace(marginal) - 1) <= input_tolerance)
        valid = valid & (jnp.linalg.eigvalsh(marginal).min() >= -input_tolerance)
    identity_n = jnp.eye(n, dtype=cost.dtype)
    identity_m = jnp.eye(m, dtype=cost.dtype)
    if state is None:
        initial = _hermitian(-cost / epsilon)
        state = QuantumDykstraState(
            initial,
            jnp.zeros_like(cost),
            jnp.zeros_like(cost),
            jnp.zeros_like(cost),
            jnp.asarray(0, jnp.int32),
        )
    else:
        if any(x.shape != cost.shape for x in state[:4]):
            raise ValueError("resume matrices must match cost shape")
        for matrix in state[:4]:
            valid = valid & jnp.isfinite(matrix).all()
    count = jnp.asarray(state.iterations)
    valid = valid & jnp.isfinite(count) & (count >= 0) & (count == jnp.floor(count))
    stop = count.astype(jnp.int32) + budget.astype(jnp.int32)
    valid = valid & (stop >= count)
    state = state._replace(iterations=count.astype(jnp.int32))

    def error_at(coupling):
        # At zero work, the initial -C/epsilon need not be PSD.
        return jnp.maximum(
            _marginal_error(coupling, marginal_a, marginal_b),
            jnp.maximum(0.0, -jnp.linalg.eigvalsh(coupling).min()),
        )

    def condition(carry):
        current, error = carry
        return (
            valid
            & jnp.isfinite(error)
            & (current.iterations < stop)
            & (error > tolerance)
        )

    def body(carry):
        current, _ = carry
        coupling, corr_a, corr_b, corr_psd, iteration = current
        candidate_a = _hermitian(coupling + corr_a)
        residual_a = marginal_a - partial_trace_second(candidate_a, n, m)
        projected_a = _hermitian(candidate_a + _lift_first(residual_a, identity_m) / m)
        new_corr_a = candidate_a - projected_a
        candidate_b = _hermitian(projected_a + corr_b)
        residual_b = marginal_b - partial_trace_first(candidate_b, n, m)
        projected_b = _hermitian(candidate_b + _lift_second(identity_n, residual_b) / n)
        new_corr_b = candidate_b - projected_b
        candidate_psd = _hermitian(projected_b + corr_psd)
        projected_psd = _project_psd(candidate_psd)
        new_corr_psd = candidate_psd - projected_psd
        next_state = QuantumDykstraState(
            projected_psd, new_corr_a, new_corr_b, new_corr_psd, iteration + 1
        )
        # PSD projection already enforces positivity; avoid another eigh per sweep.
        return next_state, _marginal_error(projected_psd, marginal_a, marginal_b)

    final, error = jax.lax.while_loop(
        condition, body, (state, error_at(state.coupling))
    )
    status = jnp.where(
        ~valid,
        INVALID_INPUT,
        jnp.where(
            ~jnp.isfinite(error),
            NUMERICAL_FAILURE,
            jnp.where(error <= tolerance, CONVERGED, ITERATION_LIMIT),
        ),
    )
    return final, SolverDiagnostics(error, final.iterations, status)
