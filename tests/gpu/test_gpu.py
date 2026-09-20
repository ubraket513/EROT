from __future__ import annotations

import jax
import numpy as np
import pytest

import erot

GPU_AVAILABLE = any(device.platform == "gpu" for device in jax.devices())


@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA-enabled JAX is unavailable")
@pytest.mark.parametrize(("dtype", "atol"), [("float64", 1e-8), ("float32", 1e-5)])
def test_cpu_gpu_classical_agreement(dtype: str, atol: float) -> None:
    cost = np.array([[0.0, 1.0], [1.0, 0.0]])
    marginals = [np.array([0.4, 0.6]), np.array([0.5, 0.5])]

    def run(device: str) -> erot.SolveResult:
        return erot.solve(
            cost,
            marginals,
            problem="classical",
            regularizer="shannon",
            method="sinkhorn",
            config=erot.SolverConfig(
                epsilon=0.5,
                tolerance=atol / 10,
                max_iterations=10_000,
                dtype=dtype,  # type: ignore[arg-type]
                device=device,
            ),
        )

    cpu, gpu = run("cpu"), run("gpu")
    assert cpu.converged and gpu.converged
    np.testing.assert_allclose(cpu.coupling, gpu.coupling, atol=atol, rtol=atol)


@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA-enabled JAX is unavailable")
def test_auto_preserves_explicit_cpu_and_gpu_arrays():
    for device in (jax.devices("cpu")[0], jax.devices("gpu")[0]):
        cost = jax.device_put(np.array([[0.0, 1.0], [1.0, 0.0]]), device)
        marginal = jax.device_put(np.array([0.4, 0.6]), device)
        result = erot.solve(
            cost,
            [marginal, marginal],
            problem="classical",
            regularizer="shannon",
            method="sinkhorn",
            config=erot.SolverConfig(0.5, device="auto"),
        )
        assert result.converged
        assert result.coupling.devices() == {device}


@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA-enabled JAX is unavailable")
@pytest.mark.parametrize(("dtype", "tolerance"), [("float64", 1e-8), ("float32", 1e-5)])
def test_cpu_gpu_quantum_entropy_agreement(dtype, tolerance):
    a = np.array([[0.6, 0.1j], [-0.1j, 0.4]])
    b = np.array([[0.45, 0.07], [0.07, 0.55]])
    results = []
    for device in ("cpu", "gpu"):
        result = erot.solve(
            np.zeros((4, 4)),
            [a, b],
            problem="quantum",
            regularizer="von_neumann",
            method="dual",
            config=erot.SolverConfig(
                0.5, tolerance=tolerance, dtype=dtype, device=device
            ),
        )
        assert result.converged
        assert next(iter(result.coupling.devices())).platform == device
        np.testing.assert_allclose(result.coupling, np.kron(a, b), atol=5 * tolerance)
        results.append(result)
    np.testing.assert_allclose(
        results[0].coupling, results[1].coupling, atol=5 * tolerance
    )


@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA-enabled JAX is unavailable")
@pytest.mark.parametrize(("dtype", "tolerance"), [("float64", 1e-8), ("float32", 1e-5)])
def test_cpu_gpu_blocked_transport_agreement(dtype, tolerance):
    from functools import partial

    from erot.geometry import PointCloudGeometry
    from erot.geometry.plan import apply_transport
    from erot.solvers.blocked_sinkhorn import solve_blocked_sinkhorn

    jax.config.update("jax_enable_x64", dtype == "float64")
    x, y = (
        np.linspace(0, 1, 5, dtype=dtype)[:, None],
        np.linspace(0, 1, 7, dtype=dtype)[:, None],
    )
    a, b = np.full(5, 0.2, dtype=dtype), np.full(7, 1 / 7, dtype=dtype)
    results = []
    for platform in ("cpu", "gpu"):
        device = jax.devices(platform)[0]
        geometry = PointCloudGeometry(
            jax.device_put(x, device), jax.device_put(y, device)
        )
        marginals = (jax.device_put(a, device), jax.device_put(b, device))
        state, diag = jax.jit(partial(solve_blocked_sinkhorn, block_size=4))(
            geometry, marginals, 0.2, tolerance, 10000
        )
        assert int(diag.status) == 0
        assert state.potentials[0].devices() == {device}
        result = apply_transport(
            geometry, state.potentials, 0.2, jax.device_put(y, device), block_size=4
        )
        results.append(np.asarray(result))
    np.testing.assert_allclose(*results, atol=5 * tolerance)
