# SPDX-License-Identifier: Apache-2.0
"""Dense trace-one von Neumann entropy QOT with complete device resume state."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..operators.quantum import gibbs_state, partial_trace_adjoint, partial_traces
from .state import CONVERGED, INVALID_INPUT, ITERATION_LIMIT, NUMERICAL_FAILURE

LINE_SEARCH_FAILED = 5


class QuantumEntropyState(NamedTuple):
    """Duals in cost units, recovered coupling and additional-work continuation."""

    dual_a: jax.Array
    dual_b: jax.Array
    coupling: jax.Array
    iterations: jax.Array
    step_size: jax.Array


class QuantumEntropyDiagnostics(NamedTuple):
    """Last-iterate objectives, signed gap and maximum marginal entry residual."""

    primal: jax.Array
    dual: jax.Array
    gap: jax.Array
    feasibility: jax.Array
    error: jax.Array
    iterations: jax.Array
    status: jax.Array


def _inner(a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.real(jnp.vdot(a, b))


def _gauge(a: jax.Array) -> jax.Array:
    a = (a + a.conj().T) / 2
    return a - jnp.trace(a).real / a.shape[0] * jnp.eye(a.shape[0], dtype=a.dtype)


def entropy_dual(
    cost: jax.Array,
    marginal_a: jax.Array,
    marginal_b: jax.Array,
    epsilon: jax.Array,
    duals: tuple[jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array, tuple[jax.Array, jax.Array]]:
    """Evaluate the trace-one dual, Gibbs coupling and Hermitian gradient.

    The primal entropy is Tr(G(log G-I)); its trace-one conjugate contributes
    epsilon*(log Tr exp(Y/epsilon)+1), including the additive constant.
    Requires valid Hermitian inputs and positive epsilon.
    """
    lifted = partial_trace_adjoint(duals)
    coupling, logz = gibbs_state(lifted - cost, epsilon)
    value = (
        _inner(marginal_a, duals[0])
        + _inner(marginal_b, duals[1])
        - epsilon * (logz + 1)
    )
    a, b = partial_traces(coupling, (marginal_a.shape[0], marginal_b.shape[0]))
    return value, coupling, (marginal_a - a, marginal_b - b)


def solve_quantum_entropy(
    cost: jax.Array,
    marginal_a: jax.Array,
    marginal_b: jax.Array,
    epsilon: jax.Array,
    tolerance: jax.Array,
    max_iterations: jax.Array,
    *,
    state: QuantumEntropyState | None = None,
    learning_rate: jax.Array = 1.0,
) -> tuple[QuantumEntropyState, QuantumEntropyDiagnostics]:
    """Solve dense entropy QOT for positive-definite trace-one marginals.

    Uses Hermitian dual gradient ascent with 30-trial Armijo backtracking.
    Budget is additional accepted updates, including zero. Resume requires the
    identical problem and controls. Trace normalization alone is not convergence:
    both marginal residual and absolute primal-dual gap must meet tolerance.
    No global dtype/device configuration or history allocation occurs here.
    """
    cost, a, b = map(jnp.asarray, (cost, marginal_a, marginal_b))
    if any(
        x.ndim != 2 or x.shape[0] != x.shape[1] or not x.shape[0] for x in (cost, a, b)
    ):
        raise ValueError("quantum cost and marginals must be nonempty square matrices")
    n, m = a.shape[0], b.shape[0]
    if cost.shape != (n * m, n * m):
        raise ValueError(
            "cost dimension must equal the product of subsystem dimensions"
        )
    dtype = jnp.result_type(cost, a, b, jnp.float32)
    cost, a, b = (x.astype(dtype) for x in (cost, a, b))
    real_dtype = cost.real.dtype
    eps, tol, rate = (
        jnp.asarray(x, real_dtype) for x in (epsilon, tolerance, learning_rate)
    )
    budget = jnp.asarray(max_iterations)
    domain_tol = 1e-6 if real_dtype == jnp.float32 else 1e-10
    valid = jnp.asarray(True)
    for matrix in (cost, a, b):
        valid = valid & jnp.isfinite(matrix).all()
        valid = valid & jnp.allclose(
            matrix, matrix.conj().T, rtol=domain_tol, atol=domain_tol
        )
    for marginal in (a, b):
        valid = valid & jnp.isclose(jnp.trace(marginal), 1, rtol=0, atol=domain_tol)
        valid = valid & (jnp.linalg.eigvalsh(marginal).min() > 0)
    for scalar in (eps, tol, rate):
        valid = valid & jnp.isfinite(scalar) & (scalar > 0)
    valid = valid & jnp.isfinite(budget) & (budget >= 0) & (budget <= 2**31 - 1)
    valid = valid & (budget == jnp.floor(budget))
    if state is None:
        state = QuantumEntropyState(
            jnp.zeros_like(a),
            jnp.zeros_like(b),
            jnp.eye(n * m, dtype=dtype) / (n * m),
            jnp.asarray(0, jnp.int32),
            rate,
        )
    else:
        if (
            state.dual_a.shape != a.shape
            or state.dual_b.shape != b.shape
            or state.coupling.shape != cost.shape
        ):
            raise ValueError("quantum entropy resume state has incompatible shapes")
        state = state._replace(
            dual_a=state.dual_a.astype(dtype),
            dual_b=state.dual_b.astype(dtype),
            coupling=state.coupling.astype(dtype),
            step_size=jnp.asarray(state.step_size, real_dtype),
        )
    count = jnp.asarray(state.iterations)
    valid = valid & jnp.isfinite(count) & (count >= 0) & (count == jnp.floor(count))
    valid = valid & (count <= 2**31 - 1 - budget)
    valid = valid & jnp.isfinite(state.step_size) & (state.step_size > 0)
    for dual in (state.dual_a, state.dual_b):
        valid = valid & jnp.isfinite(dual).all()
        valid = valid & jnp.allclose(
            dual, dual.conj().T, rtol=domain_tol, atol=domain_tol
        )
    state = state._replace(iterations=count.astype(jnp.int32))
    stop = state.iterations + budget.astype(jnp.int32)
    # Invalid epsilon is reported below, while avoiding an unnecessary divide by zero.
    safe_eps = jnp.where(jnp.isfinite(eps) & (eps > 0), eps, 1)

    def evaluate(duals, iterations):
        dual, coupling, gradients = entropy_dual(cost, a, b, safe_eps, duals)
        # Gibbs optimality gives primal-dual = -<u,gradient_a>-<v,gradient_b>.
        # This avoids a second dense eigendecomposition merely for entropy.
        gap = -_inner(duals[0], gradients[0]) - _inner(duals[1], gradients[1])
        primal = dual + gap
        residual = jnp.maximum(
            jnp.max(jnp.abs(gradients[0])), jnp.max(jnp.abs(gradients[1]))
        )
        error = jnp.maximum(residual, jnp.abs(gap))
        finite = (
            jnp.isfinite(primal) & jnp.isfinite(dual) & jnp.isfinite(coupling).all()
        )
        error = jnp.where(finite, error, jnp.inf)
        status = jnp.where(
            ~finite,
            NUMERICAL_FAILURE,
            jnp.where(error <= tol, CONVERGED, ITERATION_LIMIT),
        )
        diag = QuantumEntropyDiagnostics(
            primal, dual, gap, residual, error, iterations, status
        )
        return coupling, diag, gradients

    coupling, diag, gradients = evaluate((state.dual_a, state.dual_b), state.iterations)
    state = state._replace(coupling=coupling)
    diag = diag._replace(status=jnp.where(valid, diag.status, INVALID_INPUT))

    def condition(carry):
        current, diagnostics, _ = carry
        return (diagnostics.status == ITERATION_LIMIT) & (current.iterations < stop)

    def update(carry):
        current, diagnostics, gradients = carry
        gradients = tuple(_gauge(g) for g in gradients)
        norm2 = sum(_inner(g, g) for g in gradients)

        def trial_condition(trial):
            attempts, _, accepted, _, _, _ = trial
            return (attempts < 30) & ~accepted

        def trial_update(trial):
            attempts, step, _, candidate, candidate_diag, candidate_gradient = trial
            u = _gauge(current.dual_a + step * gradients[0])
            v = _gauge(current.dual_b + step * gradients[1])
            recovered, next_diag, next_gradient = evaluate(
                (u, v), current.iterations + 1
            )
            slack = 16 * jnp.finfo(real_dtype).eps * (1 + jnp.abs(diagnostics.dual))
            accepted = jnp.isfinite(next_diag.error) & (
                next_diag.dual >= diagnostics.dual + 1e-4 * step * norm2 - slack
            )
            candidate = QuantumEntropyState(
                u, v, recovered, current.iterations + 1, step
            )
            return (
                attempts + 1,
                jnp.where(accepted, step, step / 2),
                accepted,
                candidate,
                next_diag,
                next_gradient,
            )

        _, _, accepted, candidate, candidate_diag, candidate_gradient = (
            jax.lax.while_loop(
                trial_condition,
                trial_update,
                (
                    jnp.asarray(0, jnp.int32),
                    current.step_size,
                    jnp.asarray(False),
                    current,
                    diagnostics,
                    gradients,
                ),
            )
        )
        failed = diagnostics._replace(
            status=jnp.asarray(LINE_SEARCH_FAILED, diagnostics.status.dtype)
        )
        return jax.lax.cond(
            accepted,
            lambda: (candidate, candidate_diag, candidate_gradient),
            lambda: (current, failed, gradients),
        )

    state, diag, _ = jax.lax.while_loop(condition, update, (state, diag, gradients))
    return state, diag
