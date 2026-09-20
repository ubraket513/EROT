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


def resource_plan(
    workers,
    device,
    *,
    cpu_budget=None,
    threads_per_worker=None,
    gpu_tokens=None,
    environ=None,
    affinity=None,
):
    """Assign disjoint CPUs and allocated GPU tokens without loading JAX.

    Visible GPU tokens are opaque: preserve UUID/MIG/ordinal identifiers exactly.
    No automatic all-node GPU discovery can escape a scheduler allocation.
    """
    environ = os.environ if environ is None else environ
    affinity = sorted(os.sched_getaffinity(0)) if affinity is None else sorted(affinity)
    if (
        not affinity
        or len(set(affinity)) != len(affinity)
        or any(
            isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0
            for cpu in affinity
        )
    ):
        raise ValueError("CPU affinity must contain distinct nonnegative core IDs")
    if device not in ("cpu", "gpu"):
        raise ValueError("device must be cpu or gpu")
    available = len(affinity)
    if environ.get("SLURM_CPUS_PER_TASK"):
        try:
            scheduler_cpus = int(environ["SLURM_CPUS_PER_TASK"])
        except ValueError as exc:
            raise ValueError("invalid scheduler CPU budget") from exc
        if scheduler_cpus <= 0:
            raise ValueError("scheduler CPU budget must be positive")
        available = min(available, scheduler_cpus)
    cpu_budget = available if cpu_budget is None else cpu_budget
    if (
        isinstance(cpu_budget, bool)
        or not isinstance(cpu_budget, int)
        or not 1 <= cpu_budget <= available
    ):
        raise ValueError("CPU budget exceeds the available allocation")
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    allocated = (
        None
        if visible is None
        else [x.strip() for x in visible.split(",") if x.strip()]
    )
    if allocated == ["-1"]:
        allocated = []
    if device == "gpu":
        tokens = allocated if gpu_tokens is None else list(gpu_tokens)
        if (
            not tokens
            or any(
                not isinstance(x, str) or not x or "," in x or x == "-1" for x in tokens
            )
            or len(set(tokens)) != len(tokens)
        ):
            raise ValueError(
                "GPU allocation requires distinct visible or explicit device tokens"
            )
        if allocated is not None and any(token not in allocated for token in tokens):
            raise ValueError("GPU token is outside CUDA_VISIBLE_DEVICES allocation")
    else:
        if gpu_tokens is not None:
            raise ValueError("GPU tokens cannot be assigned to CPU workers")
        tokens = []
    workers = (len(tokens) if device == "gpu" else 1) if workers is None else workers
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or not 1 <= workers <= cpu_budget
    ):
        raise ValueError("worker count exceeds CPU allocation")
    if device == "gpu" and workers > len(tokens):
        raise ValueError("worker count exceeds GPU allocation")
    minimum_cores = cpu_budget // workers
    if threads_per_worker is not None and (
        isinstance(threads_per_worker, bool)
        or not isinstance(threads_per_worker, int)
        or not 1 <= threads_per_worker <= minimum_cores
    ):
        raise ValueError("thread budget exceeds worker CPU allocation")
    selected = affinity[:cpu_budget]
    plan, offset = [], 0
    for worker in range(workers):
        count = minimum_cores + (worker < cpu_budget % workers)
        cpus = selected[offset : offset + count]
        offset += count
        plan.append(
            {
                "worker": worker,
                "cpus": cpus,
                "threads": count if threads_per_worker is None else threads_per_worker,
                "gpu_token": tokens[worker] if device == "gpu" else None,
            }
        )
    return plan


def worker_environment(plan, device, environ=None):
    """Environment is constructed before subprocess import/initialization."""
    env = dict(os.environ if environ is None else environ)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = str(plan["threads"])
    env["CUDA_VISIBLE_DEVICES"] = plan["gpu_token"] if device == "gpu" else ""
    env["JAX_PLATFORMS"] = "cuda" if device == "gpu" else "cpu"
    return env
