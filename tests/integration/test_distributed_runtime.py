"""Initialize virtual devices in fresh processes before any backend access."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from erot.runtime.distributed import partition_rows


def test_uneven_row_partition():
    parts = [partition_rows(7, 3, index) for index in range(3)]
    assert [(p.start, p.size, p.valid_size) for p in parts] == [
        (0, 3, 3),
        (3, 3, 3),
        (6, 3, 1),
    ]
    assert partition_rows(1, 3, 2).valid_size == 0


@pytest.mark.parametrize(
    "args", [(0, 2, 0), (7, 0, 0), (7, 2, 2), (7, 2, -1), (True, 2, 0)]
)
def test_invalid_partition(args):
    with pytest.raises(ValueError):
        partition_rows(*args)


def test_two_virtual_devices_runtime():
    path = Path(__file__).resolve().parents[1] / "distributed/runtime_probe.py"
    env = dict(os.environ) | {
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
    }
    result = subprocess.run(
        [sys.executable, str(path)], env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "runtime probe passed" in result.stdout


@pytest.mark.parametrize(
    "settings",
    [
        {"num_processes": 0},
        {"process_id": 1},
        {"num_processes": 2},
        {
            "num_processes": 2,
            "coordinator_address": "localhost:1",
            "local_device_ids": [0, 0],
        },
        {"local_device_ids": [0]},
    ],
)
def test_invalid_runtime_controls(settings):
    from erot.runtime.distributed import initialize_runtime

    with pytest.raises(ValueError):
        initialize_runtime(**settings)
