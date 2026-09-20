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
