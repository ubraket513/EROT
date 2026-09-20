"""Run independent algebra checks against an explicitly supplied QOTLib checkout."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

if __package__:
    from .audit_legacy_flow import run_check
else:
    from audit_legacy_flow import run_check


def problem(*, backend="numpy", count=2, marginals=None):
    from qotlib.core import BackendContext, JaxOps, NumpyOps
    from qotlib.qot import QOTProblem

    ctx = BackendContext(
        JaxOps() if backend == "jax" else NumpyOps(),
        dtype=np.complex128,
        enable_checks=False,
    )
    if marginals is None:
        marginals = np.tile(np.eye(2) / 2, (count, 1, 1))
    return QOTProblem(
        ctx,
        ctx.asarray(np.zeros((2**count, 2**count), dtype=complex)),
        ctx.asarray(np.asarray(marginals, dtype=complex)),
        d=2,
        N=count,
    )


def operator_audit(count):
    p = problem(count=count)
    rng = np.random.default_rng(42)
    z = rng.normal(size=(2**count, 2**count)) + 1j * rng.normal(
        size=(2**count, 2**count)
    )
    x = (z + z.conj().T) / 2
    z = rng.normal(size=(count, 2, 2)) + 1j * rng.normal(size=(count, 2, 2))
    y = (z + z.conj().swapaxes(-1, -2)) / 2
    references = []
    for keep in range(count):
        tensor = x.reshape((2,) * (2 * count))
        remaining = count
        for axis in reversed(range(count)):
            if axis != keep:
                tensor = np.trace(tensor, axis1=axis, axis2=axis + remaining)
                remaining -= 1
        references.append(tensor)
    actual = p.A.apply(x)
    error = float(np.max(np.abs(actual - np.stack(references))))
    adjoint_error = float(abs(np.vdot(actual, y).real - np.vdot(x, p.A.rapply(y)).real))
    return {
        "status": "passed" if max(error, adjoint_error) < 1e-11 else "mismatch",
        "partial_trace_error": error,
        "adjoint_error": adjoint_error,
        "subsystem_count": count,
        "seed": 42,
    }


def entropy_audit(logarithmic):
    from qotlib.regularization import EntropyReg, EntropyRegLog, SDPRegularized

    p, epsilon = problem(), 0.7
    reg = (EntropyRegLog if logarithmic else EntropyReg)(p.A.dom.ctx, epsilon)
    regularized = SDPRegularized(p, reg)
    primal = p.primal_from_array(np.eye(4, dtype=complex) / 4)
    shift = 0.0 if logarithmic else -epsilon * np.log(4) / 2
    dual = p.dual_from_array(np.tile(shift * np.eye(2, dtype=complex), (2, 1, 1)))
    primal_value = float(regularized.primal_objective_reg(primal))
    dual_value = float(regularized.dual_objective_reg(dual))
    expected = epsilon * (-np.log(4) - 1)
    return {
        "status": "passed"
        if np.allclose([primal_value, dual_value], expected, atol=1e-12)
        else "mismatch",
        "primal_value": primal_value,
        "dual_value": dual_value,
        "reference_value": float(expected),
        "epsilon": epsilon,
        "objective": "epsilon * Tr(Gamma * (log(Gamma) - I))",
    }


def normalization_audit():
    from qotlib.regularization import EntropyReg, SDPRegularized

    p = problem(marginals=np.stack([np.diag([0.2, 0.8]), np.diag([0.6, 0.4])]))
    reg = SDPRegularized(p, EntropyReg(p.A.dom.ctx, 0.7))
    primal = reg.primal_from_dual(p.dual_from_array(np.zeros((2, 2, 2), complex)))
    mass = float(np.trace(primal.X).real)
    residual = float(np.linalg.norm(p.A.apply(primal.X) - p.b))
    return {
        "status": "passed" if abs(mass - 1) < 1e-12 and residual > 0.1 else "mismatch",
        "interpretation": "trace normalization does not enforce prescribed marginals",
        "trace": mass,
        "marginal_residual": residual,
    }


def gradient_audit():
    import jax
    import jax.numpy as jnp
    from qotlib.regularization import EntropyReg, SDPRegularized

    p = problem(backend="jax")
    reg = SDPRegularized(p, EntropyReg(p.A.dom.ctx, 0.7))
    y = jnp.asarray([[[0.2, 0.1j], [-0.1j, -0.1]], [[0.3, 0.2], [0.2, -0.3]]])
    direction = jnp.asarray([[[0.3, 0.2j], [-0.2j, 0.1]], [[0.1, 0.4], [0.4, 0.2]]])

    def objective(t):
        return reg.dual_objective_reg(p.dual_from_array(y + t * direction))

    derivative = float(jax.grad(objective)(0.0))
    reference = float((objective(1e-5) - objective(-1e-5)) / 2e-5)
    return {
        "status": "passed"
        if np.isclose(derivative, reference, rtol=1e-7)
        else "mismatch",
        "autodiff": derivative,
        "finite_difference": reference,
        "step": 1e-5,
    }


def lanczos_audit(scale):
    from qotlib.core import BackendContext, NumpyOps
    from qotlib.linalg import stochastic_lanczos

    ctx = BackendContext(NumpyOps(), dtype=np.complex128, enable_checks=False)
    matrix = np.diag(np.arange(1, 9) * scale).astype(complex)
    value, vector = stochastic_lanczos(
        ctx, lambda x: matrix @ x, np.ones(8, complex), max_iter=12, tol=1e-10
    )
    norm = float(np.linalg.norm(vector))
    residual = float(np.linalg.norm(matrix @ vector - value * vector) / scale)
    return {
        "status": "passed"
        if np.isclose(value, scale, rtol=1e-8)
        and abs(norm - 1) < 1e-8
        and residual < 1e-8
        else "mismatch",
        "eigenvalue": float(np.real(value)),
        "reference_eigenvalue": scale,
        "vector_norm": norm,
        "relative_residual": residual,
    }


def optimizer_budget_audit(budget):
    import jax.numpy as jnp
    import optax
    from qotlib.regularization import EntropyReg, SDPRegularized
    from qotlib.solvers._optax import run_optax_solver

    p = problem(backend="jax")
    regularized = SDPRegularized(p, EntropyReg(p.A.dom.ctx, 0.7))
    initial = p.dual_from_array(jnp.zeros((2, 2, 2), dtype=jnp.complex128))
    result = run_optax_solver(
        regularized, initial, optax.sgd(0.01), max_iter=budget, tol=1e-10
    )
    # At zero duals each gradient block of the negative dual is 1.5 * I.
    expected = np.tile(-0.015 * budget * np.eye(2), (2, 1, 1))
    error = float(np.max(np.abs(np.asarray(result.dual.y) - expected)))
    return {
        "status": "passed" if error < 1e-12 else "mismatch",
        "iteration_budget": budget,
        "expected_dual": expected.tolist(),
        "actual_dual_real": np.asarray(result.dual.y).real.tolist(),
        "max_error": error,
        "history_length": len(result.dual_obj),
        "tol_reached": result.tol_reached,
    }


def slack_feasibility(slack, equality_residual, tolerance=1e-7):
    """Check all constraints on a candidate block dual slack, not just equality."""
    slack = np.asarray(slack)
    finite = bool(np.isfinite(slack).all() and np.isfinite(equality_residual))
    hermiticity = float(np.linalg.norm(slack - slack.conj().swapaxes(-1, -2)))
    minimum = float(np.linalg.eigvalsh(slack).min()) if finite else float("nan")
    return {
        "feasible": finite
        and hermiticity <= tolerance
        and minimum >= -tolerance
        and equality_residual <= tolerance,
        "slack_min_eigenvalue": minimum,
        "slack_hermiticity_residual": hermiticity,
        "feasibility_tolerance": tolerance,
    }


def block_audit(iterations=1000):
    import cvxpy as cp
    import jax.numpy as jnp
    from qotlib.solvers.block_qot.pdhg import PDHGStepSize, block_pdhg
    from qotlib.solvers.block_qot.problem import BlockQOT, DualU, DualV, PrimalX

    eye = jnp.eye(2, dtype=jnp.complex128)
    z = jnp.diag(jnp.array([1, -1], dtype=jnp.complex128))
    cost = jnp.diag(jnp.array([0, 1, 1, 0], dtype=jnp.complex128))
    cliques = jnp.arange(4)[None, :]
    u0 = DualU(jnp.zeros((2, 2, 2), dtype=jnp.complex128), N=2, d=2)
    v0 = DualV(jnp.zeros((1, 4, 4), dtype=jnp.complex128), cliques, N=2, d=2)
    p = BlockQOT(
        C_strings=jnp.stack([jnp.stack([eye, eye]), jnp.stack([z, z])]),
        C_coeffs=jnp.array([0.5, -0.5]),
        C_ptraces=DualU(jnp.stack([eye, eye]), N=2, d=2),
        C_blocks=DualV(cost[None, :, :], cliques, N=2, d=2),
        C_l2_norm_sq=2.0,
        marginals=DualU(jnp.stack([eye / 2, eye / 2]), N=2, d=2),
        cliques=cliques,
        d=2,
        N=2,
    )
    _, u, v, _ = block_pdhg(
        p,
        PrimalX(0.0, u0, v0, N=2, d=2),
        u0,
        v0,
        PDHGStepSize(0.2, 0.2, 0.2, 1.0),
        iterations,
    )
    regularization = 1e-4  # Fixed in the historical v_step, not configurable.
    u1, u2 = cp.Variable((2, 2), symmetric=True), cp.Variable((2, 2), symmetric=True)
    slack = cp.Variable((4, 4), PSD=True)
    ref = cp.Problem(
        cp.Maximize(
            (cp.trace(u1) + cp.trace(u2)) / 2
            - regularization / 2 * cp.sum_squares(slack)
        ),
        [
            cp.kron(u1, np.eye(2)) + cp.kron(np.eye(2), u2) + slack
            == np.asarray(cost).real
        ],
    )
    ref.solve(solver="CLARABEL")
    if ref.status != "optimal":
        raise ValueError(f"independent block reference status: {ref.status}")
    bare = float(p.dual_objective(u, v))
    penalty = float(regularization / 2 * jnp.sum(jnp.abs(v.value) ** 2))
    residual = float(jnp.linalg.norm(p.K1(u) + p.K2(v) - cost))
    value = bare - penalty
    feasibility = slack_feasibility(np.asarray(v.value), residual)
    return {
        "status": "passed"
        if feasibility["feasible"] and abs(value - ref.value) < 1e-7
        else "mismatch",
        "iterations": iterations,
        "equality_residual": residual,
        "reported_objective": bare,
        "omitted_penalty": penalty,
        "corrected_objective": value,
        "reference_objective": float(ref.value),
        **feasibility,
        "interpretation": "full-clique dual with squared-slack regularization 1e-4",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = args.source_path.resolve()
    if not (source / "qotlib").is_dir():
        parser.error("--source-path must contain qotlib")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    import jax

    jax.config.update("jax_enable_x64", True)
    module = importlib.import_module("qotlib")
    if not Path(module.__file__).resolve().is_relative_to(source):
        raise ValueError("imported qotlib is outside the specified checkout")
    checks = [
        run_check("complex_operator_N2", lambda: operator_audit(2)),
        run_check("complex_operator_N3", lambda: operator_audit(3)),
        run_check("entropy_exponential_conjugate", lambda: entropy_audit(False)),
        run_check("entropy_log_conjugate", lambda: entropy_audit(True)),
        run_check("normalization_limitation", normalization_audit),
        run_check("complex_dual_directional_derivative", gradient_audit),
        run_check("lanczos_unit_scale", lambda: lanczos_audit(1.0)),
        run_check("lanczos_large_scale", lambda: lanczos_audit(1e12)),
        run_check("optimizer_zero_budget", lambda: optimizer_budget_audit(0)),
        run_check("optimizer_one_budget", lambda: optimizer_budget_audit(1)),
        run_check("block_pdhg_full_clique", block_audit),
        run_check("block_pdhg_full_clique_3000", lambda: block_audit(3000)),
    ]
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    report = {
        "source_path": str(source),
        "source_revision": revision,
        "versions": {"jax": jax.__version__, "numpy": np.__version__},
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return int(any(check["status"] != "passed" for check in checks))


if __name__ == "__main__":
    raise SystemExit(main())
