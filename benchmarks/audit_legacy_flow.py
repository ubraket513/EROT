"""Characterize an explicitly supplied historical jko_lab checkout."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import subprocess
import sys
from functools import partial
from pathlib import Path

import numpy as np


def reference_value(cost, a, b, epsilon):
    import cvxpy as cp

    plan = cp.Variable(cost.shape, nonneg=True)
    problem = cp.Problem(
        cp.Minimize(
            cp.sum(cp.multiply(cost, plan))
            - epsilon * cp.sum(cp.entr(plan))
            - epsilon * cp.sum(plan)
        ),
        [cp.sum(plan, axis=1) == a, cp.sum(plan, axis=0) == b],
    )
    problem.solve(solver="CLARABEL")
    if problem.status != "optimal":
        raise ValueError(f"independent reference status: {problem.status}")
    return float(problem.value)


def derivative_audit(predicted, reference, *, rtol=2e-3, atol=1e-4):
    valid = bool(np.isfinite(predicted) and np.isfinite(reference))
    matches = valid and bool(np.isclose(predicted, reference, rtol=rtol, atol=atol))
    return {
        "status": "passed" if matches else "mismatch",
        "predicted": float(predicted) if np.isfinite(predicted) else None,
        "reference": float(reference) if np.isfinite(reference) else None,
        "rtol": rtol,
        "atol": atol,
    }


def audit_gradient(legacy):
    import jax.numpy as jnp

    cost = np.array([[0.0, 0.7], [0.3, 0.1], [0.9, 0.2]])
    a, b = np.array([0.2, 0.5, 0.3]), np.array([0.4, 0.6])
    direction, h, epsilon = np.array([1.0, -1.0, 0.0]), 1e-3, 0.7
    solver = legacy.SinkhornJKO(
        C=jnp.asarray(cost),
        rho0=jnp.asarray(b),
        eta=0.1,
        epsilon=epsilon,
        sinkhorn_iters=10000,
        tol=1e-10,
    )
    gradient, potential, _, iterations = solver.compute_W2_gradient(
        jnp.asarray(a), jnp.asarray(b)
    )
    derivative = (
        reference_value(cost, a + h * direction, b, epsilon)
        - reference_value(cost, a - h * direction, b, epsilon)
    ) / (2 * h)
    return {
        **derivative_audit(np.dot(np.asarray(gradient), direction), derivative),
        "potential_derivative": float(np.dot(np.asarray(potential), direction)),
        "iterations": int(iterations),
        "input": {
            "cost": cost.tolist(),
            "a": a.tolist(),
            "b": b.tolist(),
            "epsilon": epsilon,
            "direction": direction.tolist(),
            "h": h,
        },
    }


def audit_pdhg(legacy, mode):
    import cvxpy as cp
    import jax.numpy as jnp

    x = np.array([0.0, 0.5, 1.0])
    cost = (x[:, None] - x[None, :]) ** 2
    previous = np.array([0.2, 0.5, 0.3])
    target = np.array([0.3, 0.3, 0.4])
    eta, iterations = 0.1, 2000
    plan_ref = cp.Variable((3, 3), nonneg=True)
    rho_ref = cp.sum(plan_ref, axis=1)
    reference = cp.Problem(
        cp.Minimize(
            eta * 0.5 * cp.sum_squares(rho_ref - target)
            + 0.5 * cp.sum(cp.multiply(cost, plan_ref))
        ),
        [cp.sum(plan_ref, axis=0) == previous],
    )
    reference.solve(solver="CLARABEL")
    if reference.status != "optimal":
        raise ValueError(f"independent PDHG reference: {reference.status}")
    prox = partial(legacy.proxF_quadratic, b=jnp.asarray(target), lam=1.0)
    step_options = {} if mode == "class-default" else {"tau": 0.01, "sigma": 0.01}
    if mode == "functional":
        state = legacy.pdhg_jko(
            jnp.asarray(cost),
            jnp.asarray(previous),
            eta,
            prox,
            num_iters=iterations,
            **step_options,
        )
        rho, plan = np.asarray(state.rho), np.asarray(state.pi)
        rho_kind = "raw iterate"
    else:
        solver = legacy.PrimalDualJKO(
            C=jnp.asarray(cost),
            rho0=jnp.asarray(previous),
            eta=eta,
            proxF=prox,
            inner_steps=iterations,
            **step_options,
        )
        rho, plan, _, _ = solver.take_step(jnp.asarray(previous))
        rho, plan = np.asarray(rho), np.asarray(plan)
        rho_kind = "normalized class return; raw iterate unavailable"
    row_error = float(np.linalg.norm(plan.sum(axis=1) - rho))
    col_error = float(np.linalg.norm(plan.sum(axis=0) - previous))
    value = float(eta * 0.5 * np.sum((rho - target) ** 2) + 0.5 * np.sum(cost * plan))
    finite = bool(np.isfinite(rho).all() and np.isfinite(plan).all())
    matches = (
        finite
        and float(rho.min()) >= -1e-9
        and float(plan.min()) >= -1e-9
        and max(row_error, col_error) < 1e-5
        and np.isclose(value, reference.value, atol=1e-5, rtol=1e-4)
    )
    return {
        "status": "passed" if matches else "mismatch",
        "rho_kind": rho_kind,
        "mass": float(rho.sum()),
        "minimum_density": float(rho.min()),
        "minimum_coupling": float(plan.min()),
        "nonnegativity_tolerance": 1e-9,
        "objective": value,
        "reference_objective": float(reference.value),
        "row_residual": row_error,
        "column_residual": col_error,
        "rho": rho.tolist(),
        "input": {
            "cost": cost.tolist(),
            "previous": previous.tolist(),
            "target": target.tolist(),
            "time_step": eta,
            "iterations": iterations,
            **step_options,
        },
    }


def audit_entropy_prox(legacy, alpha):
    import jax.numpy as jnp
    from scipy.optimize import brentq

    z = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
    got = np.asarray(legacy.proxF_entropy(jnp.asarray(z), alpha))
    if alpha == 0:
        expected = np.maximum(z, 0.0)
    else:
        expected = np.array(
            [
                np.exp(
                    brentq(
                        lambda y: np.exp(y) - value + alpha * y,
                        min(value / alpha - 1.0, np.log(alpha) - 1.0, -1.0),
                        max(np.log(max(value, 1.0)) + 1.0, 1.0),
                    )
                )
                for value in z
            ]
        )
    finite = bool(np.isfinite(got).all())
    return {
        "status": "passed"
        if finite and np.allclose(got, expected, rtol=1e-6, atol=1e-9)
        else "mismatch",
        "input": {"z": z.tolist(), "alpha": alpha},
        "actual": [float(v) if np.isfinite(v) else None for v in got],
        "reference": expected.tolist(),
    }


def run_check(name, function):
    try:
        nonfinite = False

        def clean(value):
            nonlocal nonfinite
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                nonfinite = True
                return None
            if isinstance(value, dict):
                return {key: clean(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [clean(item) for item in value]
            return value

        record = clean({"check": name, **function()})
        if nonfinite and record.get("status") == "passed":
            record["status"] = "mismatch"
        return record
    except ModuleNotFoundError as exc:
        return {
            "check": name,
            "status": "unavailable",
            "exception": type(exc).__name__,
            "message": str(exc),
        }
    except Exception as exc:
        return {
            "check": name,
            "status": "error",
            "exception": type(exc).__name__,
            "message": str(exc),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_path.resolve()
    if not (source / "src/jko_lab").is_dir():
        parser.error("--source-path must contain src/jko_lab")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source / "src"))

    def perform():
        import jax

        jax.config.update("jax_enable_x64", True)
        legacy = importlib.import_module("jko_lab")
        if not Path(legacy.__file__).resolve().is_relative_to(source):
            raise ValueError("imported jko_lab is outside the specified source")
        return legacy

    try:
        legacy = perform()
    except Exception as exc:
        records = [
            {
                "check": "source_import",
                "status": "unavailable"
                if isinstance(exc, ModuleNotFoundError)
                else "error",
                "exception": type(exc).__name__,
                "message": str(exc),
            }
        ]
    else:
        records = [run_check("potential_derivative", lambda: audit_gradient(legacy))]
        for mode in ("class-default", "class-explicit", "functional"):
            records.append(
                run_check(f"pdhg_{mode}", lambda mode=mode: audit_pdhg(legacy, mode))
            )
        for alpha in (0.0, 1e-4, 0.1, 1.0):
            records.append(
                run_check(
                    f"entropy_prox_{alpha}",
                    lambda alpha=alpha: audit_entropy_prox(legacy, alpha),
                )
            )
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    versions = {}
    for package in ("jax", "numpy", "scipy", "cvxpy", "flax", "optax"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report = {
        "source_path": str(source),
        "source_revision": revision.stdout.strip(),
        "versions": versions,
        "checks": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return int(any(record["status"] != "passed" for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
