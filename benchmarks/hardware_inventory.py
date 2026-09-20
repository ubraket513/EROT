"""Record allocation, device visibility and NVIDIA topology without inferring capacity."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


def command_report(arguments):
    if shutil.which(arguments[0]) is None:
        return {
            "status": "unavailable",
            "command": arguments,
            "reason": "executable not found",
        }
    try:
        result = subprocess.run(
            arguments, capture_output=True, text=True, timeout=20, check=False
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "command": arguments}
    return {
        "status": "available" if result.returncode == 0 else "error",
        "command": arguments,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def read_optional(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def inventory(backend):
    # NVIDIA inventory precedes JAX allocation; free memory is a point-in-time value.
    report = {
        "schema_version": 1,
        "hostname": platform.node(),
        "system": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "host_meminfo": read_optional("/proc/meminfo"),
        "cgroup_memory_limit": read_optional("/sys/fs/cgroup/memory.max"),
        "cgroup_cpu_quota": read_optional("/sys/fs/cgroup/cpu.max"),
        "allocation": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NUM_NODES",
                "SLURM_NTASKS",
                "SLURM_PROCID",
                "SLURM_LOCALID",
                "SLURM_CPUS_PER_TASK",
                "SLURM_JOB_GPUS",
                "SLURM_STEP_GPUS",
                "CUDA_VISIBLE_DEVICES",
                "JAX_PLATFORMS",
                "OMP_NUM_THREADS",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_PYTHON_CLIENT_MEM_FRACTION",
            )
        },
        "nvidia_query": command_report(["nvidia-smi", "-q", "-x"]),
        "nvidia_topology": command_report(["nvidia-smi", "topo", "-m"]),
        "infiniband": command_report(["ibv_devinfo"]),
        "requested_backend": backend,
    }
    try:
        import jax

        devices = jax.devices(backend)
        report["jax"] = {
            "status": "available",
            "version": jax.__version__,
            "process_index": jax.process_index(),
            "process_count": jax.process_count(),
            "devices": [
                {
                    "id": d.id,
                    "platform": d.platform,
                    "kind": d.device_kind,
                    "process_index": d.process_index,
                    "memory_stats": d.memory_stats(),
                }
                for d in devices
            ],
        }
    except Exception as exc:
        report["jax"] = {
            "status": "unavailable",
            "exception": type(exc).__name__,
            "reason": str(exc),
        }
    report["interpretation"] = (
        "Inventory only, not a performance or support certification. Read NVIDIA XML for "
        "driver, MIG, memory and sharing details. Topology does not prove collective bandwidth. "
        "Host RAM may exceed the job cgroup limit. GPU free memory was sampled before JAX initialization."
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inventory(args.backend)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return int(report["jax"]["status"] != "available")


if __name__ == "__main__":
    raise SystemExit(main())
