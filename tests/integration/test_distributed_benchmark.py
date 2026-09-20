"""Isolated fresh-process scaling reports with matched scientific controls."""

import json
import os
import subprocess
import sys
from pathlib import Path


def test_one_two_device_benchmark_reports(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "kind": "classical",
                "backend": "blocked",
                "n": 33,
                "m": 41,
                "features": 2,
                "epsilon": 0.3,
                "tolerance": 1e-8,
                "block_size": 8,
                "max_iterations": 1000,
            }
        )
    )
    script = Path(__file__).resolve().parents[2] / "benchmarks/distributed_sinkhorn.py"
    reports = []
    for devices in (1, 2):
        root = tmp_path / f"devices-{devices}"
        env = dict(os.environ) | {
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}",
        }
        process = subprocess.run(
            [
                sys.executable,
                str(script),
                str(config),
                "--output-directory",
                str(root),
                "--device",
                "cpu",
                "--repeats",
                "2",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert process.returncode == 0, process.stdout + process.stderr
        report = json.loads((root / "report.json").read_text())
        assert report["valid"]
        assert len(report["warm_seconds"]) == 2
        assert report["error"] <= 1e-8
        assert report["topology"]["devices"] == devices
        assert report["ranks"][0]["compiler_memory"]["full_matrix_shapes"] == []
        assert report["communication_share"] is None
        assert all(
            value["peak_bytes_in_use"] is None
            for value in report["ranks"][0]["device_memory"]
        )
        reports.append(report)
    assert reports[0]["case_digest"] == reports[1]["case_digest"]


def test_two_process_benchmark_report(tmp_path):
    import socket

    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "kind": "classical",
                "backend": "blocked",
                "n": 7,
                "m": 5,
                "epsilon": 0.3,
                "tolerance": 1e-8,
                "max_iterations": 1000,
                "block_size": 3,
            }
        )
    )
    root = tmp_path / "two-process"
    script = Path(__file__).resolve().parents[2] / "benchmarks/distributed_sinkhorn.py"
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
                str(script),
                str(config),
                "--output-directory",
                str(root),
                "--repeats",
                "2",
                "--coordinator",
                address,
                "--processes",
                "2",
                "--process-id",
                str(rank),
                "--local-device-ids",
                "0",
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
        assert all(process.returncode == 0 for process in processes), output
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    report = json.loads((root / "report.json").read_text())
    assert report["valid"] and report["topology"]["processes"] == 2
    assert [rank["rank"] for rank in report["ranks"]] == [0, 1]
