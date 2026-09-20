"""Real local multiprocess collectives with bounded subprocess cleanup."""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_two_cpu_processes_match_reference(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{listener.getsockname()[1]}"
    env = {
        key: value
        for key, value in os.environ.items()
        if key.lower() not in ("http_proxy", "https_proxy", "all_proxy")
    }
    env.update(
        JAX_PLATFORMS="cpu",
        XLA_FLAGS="--xla_force_host_platform_device_count=1",
        JAX_CPU_COLLECTIVES_IMPLEMENTATION="gloo",
    )
    script = Path(__file__).resolve().parents[1] / "distributed/process_probe.py"
    processes = []
    try:
        for rank in range(2):
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        str(script),
                        address,
                        str(rank),
                        str(tmp_path / f"rank{rank}.json"),
                    ],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )
        outputs = [process.communicate(timeout=60)[0] for process in processes]
        assert all(process.returncode == 0 for process in processes), "\n".join(outputs)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    records = [
        json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)
    ]
    assert records[0]["status"] == records[1]["status"] == 0
    assert records[0]["iterations"] == records[1]["iterations"]
    assert records[0]["error"] == records[1]["error"]
    np.testing.assert_array_equal(records[0]["g"], records[1]["g"])
    f = np.concatenate([record["f"] for record in records])
    g = np.asarray(records[0]["g"])
    cost = (np.arange(8)[:, None] / 7 - np.linspace(0, 1, 5)[None, :]) ** 2
    plan = np.exp((f[:, None] + g[None, :] - cost) / 0.2)
    np.testing.assert_allclose(plan.sum(1), np.r_[np.full(7, 1 / 7), 0], atol=1e-10)
    np.testing.assert_allclose(plan.sum(0), np.full(5, 0.2), atol=1e-10)

    import jax
    import jax.numpy as jnp

    from erot.solvers.sinkhorn import solve_sinkhorn

    jax.config.update("jax_enable_x64", True)
    reference, diagnostics = solve_sinkhorn(
        jnp.asarray(cost),
        (jnp.asarray(np.r_[np.full(7, 1 / 7), 0]), jnp.full(5, 0.2)),
        0.2,
        1e-10,
        2000,
    )
    assert int(diagnostics.status) == 0
    np.testing.assert_allclose(f, reference.potentials[0], atol=1e-9, rtol=0)
    np.testing.assert_allclose(g, reference.potentials[1], atol=1e-9, rtol=0)
