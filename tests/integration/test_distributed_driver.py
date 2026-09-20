"""Installed-style two-process driver, exact restart and configuration agreement."""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np


def run_group(config_paths, root, *options):
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
    processes = []
    try:
        for rank in range(2):
            command = [
                sys.executable,
                "-m",
                "erot.distributed",
                str(config_paths[rank]),
                "--run-directory",
                str(root),
                "--coordinator",
                address,
                "--processes",
                "2",
                "--process-id",
                str(rank),
                *options,
            ]
            processes.append(
                subprocess.Popen(
                    command,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )
        output = [process.communicate(timeout=60)[0] for process in processes]
        return [p.returncode for p in processes], output
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()


def write_config(tmp_path, **changes):
    config = {
        "name": "distributed-test",
        "kind": "classical",
        "backend": "blocked",
        "n": 7,
        "m": 5,
        "features": 2,
        "seed": 2,
        "dtype": "float64",
        "device": "cpu",
        "epsilon": 0.2,
        "tolerance": 1e-9,
        "max_iterations": 1000,
        "chunk_size": 3,
        "block_size": 3,
        **changes,
    }
    path = tmp_path / f"config-{len(list(tmp_path.glob('config-*.json')))}.json"
    path.write_text(json.dumps(config))
    return path


def payloads(root):
    generation = (
        root / "checkpoints" / (root / "checkpoints/LATEST").read_text().strip()
    )
    manifest = json.loads((generation / "manifest.json").read_text())
    result = []
    for rank, item in enumerate(manifest["ranks"]):
        with np.load(
            generation / f"rank-{rank:06d}" / item["generation"] / "arrays.npz"
        ) as arrays:
            result.append({key: arrays[key].copy() for key in arrays.files})
    return result


def test_driver_two_process_restart(tmp_path):
    config = write_config(tmp_path)
    whole, resumed = tmp_path / "whole", tmp_path / "resumed"
    codes, output = run_group([config, config], whole)
    assert codes == [0, 0], output
    codes, output = run_group([config, config], resumed, "--stop-after-chunks", "1")
    assert codes == [0, 0], output
    assert json.loads((resumed / "result.json").read_text())["status"] == "checkpointed"
    codes, output = run_group([config, config], resumed, "--resume")
    assert codes == [0, 0], output
    a, b = (json.loads((root / "result.json").read_text()) for root in (whole, resumed))
    assert a["status"] == b["status"] == "completed"
    assert a["iterations"] == b["iterations"]
    assert a["error"] == b["error"]
    for rank_a, rank_b in zip(payloads(whole), payloads(resumed)):
        assert rank_a.keys() == rank_b.keys()
        for key in rank_a:
            np.testing.assert_array_equal(rank_a[key], rank_b[key])
    codes, output = run_group([config, config], resumed)
    assert all(code != 0 for code in codes), output
    assert "resume" in "\n".join(output)


def test_driver_cross_rank_config_mismatch(tmp_path):
    a, b = write_config(tmp_path, seed=1), write_config(tmp_path, seed=2)
    codes, output = run_group([a, b], tmp_path / "mismatch")
    assert all(code != 0 for code in codes), output
    assert "disagree" in "\n".join(output)
    assert not (tmp_path / "mismatch/checkpoints/LATEST").exists()


def test_driver_failure_keeps_checkpoint(tmp_path):
    config = write_config(tmp_path, max_iterations=1)
    codes, output = run_group([config, config], tmp_path / "failed")
    assert codes == [1, 1], output
    result = json.loads((tmp_path / "failed/result.json").read_text())
    assert result["status"] == "failed" and result["iterations"] == 1
    assert len(payloads(tmp_path / "failed")) == 2


def test_driver_one_rank_bad_config_is_coordinated(tmp_path):
    config = write_config(tmp_path)
    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]")
    codes, output = run_group([config, invalid], tmp_path / "bad-input")
    assert all(code != 0 for code in codes), output
    assert "configuration" in "\n".join(output)


def test_driver_global_owner_lock(tmp_path):
    from erot.runtime.environment import run_lock

    config = write_config(tmp_path)
    root = tmp_path / "owned"
    with run_lock(root):
        codes, output = run_group([config, config], root)
    assert all(code != 0 for code in codes), output
    assert "owned" in "\n".join(output)


def test_driver_rejects_changed_restart_inputs(tmp_path):
    original, changed = write_config(tmp_path, seed=1), write_config(tmp_path, seed=3)
    root = tmp_path / "changed"
    codes, output = run_group([original, original], root, "--stop-after-chunks", "1")
    assert codes == [0, 0], output
    previous = (root / "checkpoints/LATEST").read_text()
    codes, output = run_group([changed, changed], root, "--resume")
    assert all(code != 0 for code in codes), output
    assert "incompatible" in "\n".join(output)
    assert (root / "checkpoints/LATEST").read_text() == previous


def test_rank_script_one_process_two_devices(tmp_path):
    config = write_config(tmp_path)
    root = tmp_path / "one-process"
    script = Path(__file__).resolve().parents[2] / "hpc/distributed_rank.sh"
    env = dict(os.environ) | {
        "EROT_CONFIG": str(config),
        "EROT_RUN_DIRECTORY": str(root),
        "EROT_DEVICE": "cpu",
        "EROT_PYTHON": sys.executable,
        "SLURM_NTASKS": "1",
        "SLURM_PROCID": "0",
        "SLURM_CPUS_PER_TASK": "2",
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
    }
    result = subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((root / "result.json").read_text())
    assert report["status"] == "completed"
    assert report["topology"]["devices"] == 2 and report["topology"]["processes"] == 1
    resources = json.loads((root / "resources-rank-000000.json").read_text())
    assert resources["thread_environment"]["OMP_NUM_THREADS"] == "2"

    codes, output = run_group([config, config], tmp_path / "two-process")
    assert codes == [0, 0], output
    other = json.loads((tmp_path / "two-process/result.json").read_text())
    assert report["iterations"] == other["iterations"]
    np.testing.assert_allclose(report["error"], other["error"], atol=1e-12, rtol=1e-6)


def test_spooled_batch_uses_explicit_shared_rank_script(tmp_path):
    import shutil

    repository = Path(__file__).resolve().parents[2]
    spool = tmp_path / "spool/job1"
    spool.mkdir(parents=True)
    batch = spool / "slurm_script"
    shutil.copyfile(repository / "hpc/run_distributed.sbatch", batch)
    binary = tmp_path / "bin"
    binary.mkdir()
    srun = binary / "srun"
    srun.write_text('#!/usr/bin/env bash\nset -euo pipefail\nshift\nexec "$@"\n')
    srun.chmod(0o755)
    config = write_config(tmp_path)
    root = tmp_path / "spooled-result"
    env = dict(os.environ) | {
        "PATH": str(binary) + os.pathsep + os.environ["PATH"],
        "EROT_CONFIG": str(config),
        "EROT_RUN_DIRECTORY": str(root),
        "EROT_RANK_SCRIPT": str(repository / "hpc/distributed_rank.sh"),
        "EROT_DEVICE": "cpu",
        "EROT_PYTHON": sys.executable,
        "SLURM_NTASKS": "1",
        "SLURM_PROCID": "0",
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=1",
    }
    completed = subprocess.run(
        ["bash", str(batch)], env=env, capture_output=True, text=True, timeout=60
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads((root / "result.json").read_text())["status"] == "completed"
