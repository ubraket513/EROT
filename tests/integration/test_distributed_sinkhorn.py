"""Collective correctness on two virtual CPU devices, not GPU performance."""

import os
import subprocess
import sys
from pathlib import Path


def test_distributed_sinkhorn_probe():
    path = Path(__file__).resolve().parents[1] / "distributed/solver_probe.py"
    env = dict(os.environ) | {
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
    }
    result = subprocess.run(
        [sys.executable, str(path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "collective solver probe passed" in result.stdout
    print(result.stdout)
