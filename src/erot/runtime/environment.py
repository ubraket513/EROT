"""Host resource metadata and exclusive run ownership without backend startup."""

import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def run_lock(path):
    """Hold a nonblocking advisory lock; stable lock file avoids unlink races."""
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "experiment directory is already owned by another worker"
            ) from exc
        try:
            stream.seek(0)
            stream.truncate()
            json.dump({"pid": os.getpid(), "host": platform.node()}, stream)
            stream.flush()
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def runtime_versions():
    result = {"python": platform.python_version()}
    for name in (
        "EROT",
        "jax",
        "jaxlib",
        "numpy",
        "jax-cuda12-plugin",
        "jax-cuda13-plugin",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def source_identity():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        revision = completed.stdout.strip() if completed.returncode == 0 else None
    except OSError:
        revision = None
    return {
        "digest": digest.hexdigest(),
        "revision": revision,
    }


def observed_resources():
    return {
        "affinity": sorted(os.sched_getaffinity(0)),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "XLA_FLAGS",
            )
        },
        "slurm": {
            key: os.environ.get(key)
            for key in ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_CPUS_PER_TASK")
        },
    }
