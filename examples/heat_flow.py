"""Compare a discrete entropy JKO trajectory with a Neumann heat mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from erot.flows import Entropy, initialize_flow, run_flow_chunk


def run(
    cells=16,
    time_step=0.02,
    steps=2,
    backend="pdhg",
    epsilon=0.002,
    tolerance=1e-7,
    max_iterations=30000,
    inner_tolerance=1e-10,
):
    """Use cell centers on [0,1], zero-flux boundaries and density 1+.2*cos(pi*x)."""
    jax.config.update("jax_enable_x64", True)
    x = (jnp.arange(cells, dtype=jnp.float64) + 0.5) / cells
    volume = jnp.full(cells, 1 / cells)
    rho = (1 + 0.2 * jnp.cos(jnp.pi * x)) * volume
    cost = (x[:, None] - x[None, :]) ** 2
    energy = Entropy(volume)
    state = initialize_flow(rho, backend=backend)
    options = {"inner_tolerance": inner_tolerance} if backend == "sinkhorn" else {}
    result = run_flow_chunk(
        state,
        cost,
        energy,
        time_step,
        tolerance,
        max_iterations,
        steps=steps,
        backend=backend,
        epsilon=epsilon,
        snapshot_stride=1,
        **options,
    )
    jax.block_until_ready(result)
    time = float(result.state.time)
    reference = (1 + 0.2 * jnp.exp(-(jnp.pi**2) * time) * jnp.cos(jnp.pi * x)) * volume

    def finite(value):
        value = float(value)
        return value if np.isfinite(value) else None

    return {
        "study": "Neumann heat mode on [0,1]",
        "boundary": "zero flux",
        "quadrature": "uniform cell centers; states are cell masses",
        "backend": backend,
        "cells": cells,
        "time_step": time_step,
        "requested_steps": steps,
        "accepted_steps": int(result.state.accepted_steps),
        "requested_time": steps * time_step,
        "physical_time": time,
        "epsilon": epsilon if backend == "sinkhorn" else None,
        "tolerance": tolerance,
        "inner_tolerance": inner_tolerance if backend == "sinkhorn" else None,
        "max_iterations": max_iterations,
        "status": int(result.state.status),
        "work": int(result.state.work),
        "mass_error": finite(abs(result.state.rho.sum() - rho.sum())),
        "heat_mode_l1_error": finite(jnp.sum(jnp.abs(result.state.rho - reference))),
        "initial_energy": finite(energy.value(rho)),
        "final_energy": finite(energy.value(result.state.rho)),
        "last_stationarity": finite(result.diagnostics.stationarity),
        "masses": np.asarray(result.state.rho).tolist(),
        "reference_masses": np.asarray(reference).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=16)
    parser.add_argument("--time-step", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--backend", choices=("pdhg", "sinkhorn"), default="pdhg")
    parser.add_argument("--epsilon", type=float, default=0.002)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--inner-tolerance", type=float, default=1e-10)
    parser.add_argument("--max-iterations", type=int, default=30000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.cells < 2 or args.steps < 1 or args.time_step <= 0:
        parser.error("cells>=2, steps>=1 and time_step>0 are required")
    options = vars(args).copy()
    output = options.pop("output")
    report = run(**options)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return int(report["status"] != 0 or report["accepted_steps"] != args.steps)


if __name__ == "__main__":
    raise SystemExit(main())
