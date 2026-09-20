"""Deterministic scientific workloads and bounded device chunk functions."""

import hashlib
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import xlogy

from ..flows import Entropy, Quadratic, initialize_flow, run_flow_chunk
from ..geometry import PointCloudGeometry
from ..geometry.plan import transport_objective
from ..solvers.blocked_sinkhorn import solve_blocked_sinkhorn
from ..solvers.quantum import solve_quantum_quadratic
from ..solvers.quantum_entropy import solve_quantum_entropy
from ..solvers.sinkhorn import materialize_plan, solve_sinkhorn


@dataclass
class Workload:
    initial: object
    chunk: object
    input_digest: str
    rng_state: dict
    scientific: object


def create_workload(config, device):
    """Generate only the representation required by the selected algorithm."""
    rng = np.random.default_rng(config["seed"])
    dtype = np.dtype(config["dtype"])
    n, m = config["n"], config["m"]
    inputs = []

    def array(value):
        value = np.asarray(value)
        inputs.append(value)
        return jax.device_put(value, device)

    if config["kind"] == "flow":
        x_host = (np.arange(n, dtype=dtype) + 0.5) / n
        x = array(x_host[:, None])
        if config["energy"] == "entropy":
            initial_mass = (1 + 0.2 * np.cos(np.pi * x_host)) / n
            energy = Entropy(array(np.full(n, 1 / n, dtype=dtype)))
        else:
            initial_mass = rng.uniform(0.1, 1.0, n).astype(dtype)
            initial_mass /= initial_mass.sum()
            energy = Quadratic(array(np.full(n, 1 / n, dtype=dtype)), 2.0)
        rho = array(initial_mass.astype(dtype))
        cost = (
            PointCloudGeometry(x, x)
            if config["backend"] == "blocked"
            else (x - x.T) ** 2
        )
        backend = "pdhg" if config["backend"] == "pdhg" else "sinkhorn"
        initial = initialize_flow(rho, backend=backend)

        def scientific(state):
            return {"energy": energy.value(state.rho)}

        options = {}
        if backend == "sinkhorn":
            options = {
                "epsilon": config["epsilon"],
                "inner_iterations": config["inner_iterations"],
                "inner_tolerance": config["inner_tolerance"],
                "transport_block_size": config["block_size"],
            }

        @partial(jax.jit, static_argnames=("count",))
        def chunk(state, count):
            result = run_flow_chunk(
                state,
                cost,
                energy,
                config["time_step"],
                config["tolerance"],
                config["max_iterations"],
                steps=count,
                snapshot_stride=config["snapshot_stride"],
                backend=backend,
                **options,
            )
            return (
                result.state,
                result.diagnostics,
                result.snapshots,
                result.snapshot_times,
            )
    else:
        if config["kind"] == "classical":
            x = array(rng.normal(size=(n, config["features"])).astype(dtype))
            y = array(rng.normal(size=(m, config["features"])).astype(dtype))
            a_host, b_host = rng.uniform(0.1, 1, n), rng.uniform(0.1, 1, m)
            a = array((a_host / a_host.sum()).astype(dtype))
            b = array((b_host / b_host.sum()).astype(dtype))
            if config["backend"] == "blocked":
                cost = PointCloudGeometry(x, y)
                solver = partial(
                    solve_blocked_sinkhorn, block_size=config["block_size"]
                )
            else:
                cost = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
                solver = solve_sinkhorn
            arguments = (cost, (a, b), config["epsilon"], config["tolerance"])

            def scientific(state):
                if config["backend"] == "blocked":
                    value = transport_objective(
                        cost,
                        state.potentials,
                        config["epsilon"],
                        block_size=config["block_size"],
                    )
                else:
                    plan = materialize_plan(cost, state.potentials, config["epsilon"])
                    value = jnp.sum(
                        cost * plan + config["epsilon"] * (xlogy(plan, plan) - plan)
                    )
                return {"objective": value}
        else:
            complex_dtype = np.complex128 if dtype == np.float64 else np.complex64

            def density(size):
                raw = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
                matrix = raw @ raw.conj().T + 0.2 * np.eye(size)
                return (matrix / np.trace(matrix)).astype(complex_dtype)

            a, b = array(density(n)), array(density(m))
            raw = rng.normal(size=(n * m, n * m)) + 1j * rng.normal(size=(n * m, n * m))
            cost = array(((raw + raw.conj().T) * 0.1).astype(complex_dtype))
            solver = (
                solve_quantum_entropy
                if config["backend"] == "entropy"
                else solve_quantum_quadratic
            )
            arguments = (cost, a, b, config["epsilon"], config["tolerance"])

            def scientific(state):
                coupling = state.coupling
                value = jnp.real(jnp.vdot(cost, coupling))
                if config["backend"] == "quadratic":
                    value += (
                        config["epsilon"] * jnp.real(jnp.vdot(coupling, coupling)) / 2
                    )
                else:
                    eigenvalues = jnp.maximum(jnp.linalg.eigvalsh(coupling), 0)
                    value += config["epsilon"] * jnp.sum(
                        xlogy(eigenvalues, eigenvalues) - eigenvalues
                    )
                return {"objective": value}

        initial, _ = solver(*arguments, 0)

        @jax.jit
        def chunk(state, count):
            updated, diagnostics = solver(*arguments, count, state=state)
            return updated, diagnostics, jnp.empty((0,), dtype), jnp.empty((0,), dtype)

    digest = hashlib.sha256()
    for value in inputs:
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return Workload(
        initial, chunk, digest.hexdigest(), rng.bit_generator.state, scientific
    )
