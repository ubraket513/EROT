"""Opt-in actual two-GPU numerical equivalence; never replaced by CPU probes."""

import jax
import numpy as np
import pytest
from jax.sharding import Mesh

GPUS = [device for device in jax.devices() if device.platform == "gpu"]


@pytest.mark.gpu
@pytest.mark.distributed
@pytest.mark.skipif(len(GPUS) < 2, reason="two CUDA devices are unavailable")
@pytest.mark.parametrize("dtype,tolerance", [(np.float32, 2e-5), (np.float64, 1e-10)])
def test_two_gpu_collective_reference_and_resume(dtype, tolerance):
    from erot.runtime.distributed import global_from_local
    from erot.solvers.distributed_sinkhorn import make_distributed_sinkhorn
    from erot.solvers.state import SinkhornState

    jax.config.update("jax_enable_x64", True)
    mesh = Mesh(np.asarray(GPUS[:2]), ("rows",))
    x = np.linspace(0, 1, 8, dtype=dtype)[:, None]
    y = np.linspace(-0.1, 0.9, 5, dtype=dtype)[:, None]
    a = np.array([0, 0, 0, 0, 0.2, 0.3, 0.5, 0], dtype=dtype)
    b = np.array([0.1, 0, 0.3, 0.2, 0.4], dtype=dtype)
    arguments = (
        global_from_local(x, mesh),
        global_from_local(y, mesh, True),
        global_from_local(a, mesh),
        global_from_local(b, mesh, True),
        dtype(0.2),
        dtype(tolerance),
    )
    initial = SinkhornState(
        (
            global_from_local(np.where(a > 0, 0, -np.inf).astype(dtype), mesh),
            global_from_local(np.where(b > 0, 0, -np.inf).astype(dtype), mesh, True),
        ),
        global_from_local(np.asarray(0, np.int32), mesh, True),
    )
    solve = make_distributed_sinkhorn(mesh, block_size=3)
    state, diagnostics = solve(*arguments, 2000, initial)
    assert int(diagnostics.status) == 0
    partial, _ = solve(*arguments, 2, initial)
    resumed, _ = solve(*arguments, 2000, partial)
    for actual, expected in zip(jax.tree.leaves(resumed), jax.tree.leaves(state)):
        np.testing.assert_allclose(
            actual, expected, atol=5 * tolerance, rtol=5 * tolerance
        )
    cost = (x.astype(np.float64) - y.astype(np.float64).T) ** 2
    kernel = np.exp(-cost / 0.2)
    v = np.ones(5)
    for _ in range(2000):
        u = a / (kernel @ v)
        v = b / (kernel.T @ u)
    reference = u[:, None] * kernel * v[None, :]
    f, g = map(np.asarray, state.potentials)
    actual = np.exp((f[:, None] + g[None, :] - cost) / 0.2)
    np.testing.assert_allclose(
        actual, reference, atol=5 * tolerance, rtol=5 * tolerance
    )
    assert np.all(actual[:4] == 0) and np.all(actual[-1] == 0)
