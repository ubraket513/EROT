"""Launch independent resource-isolated experiments without importing JAX."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .runtime.config import normalize_config, run_identity
from .runtime.environment import resource_plan, worker_environment

# Constant code only; CPU IDs and paths are data arguments, never interpolated code.
_BOOTSTRAP = (
    "import json,os,sys; "
    "os.sched_setaffinity(0,json.loads(sys.argv.pop(1))); "
    "from erot.experiments import main; "
    "raise SystemExit(main(sys.argv[1:]))"
)


def launch_experiments(
    configs,
    output_root,
    *,
    workers=None,
    device="cpu",
    cpu_budget=None,
    threads_per_worker=None,
    gpu_tokens=None,
    resume=False,
):
    """Run disjoint batches with one fresh process per experiment.

    A slot retains its CPU/GPU assignment across jobs. Results and logs are
    isolated by scientific run identity; duplicate identities are rejected.
    """
    configs = [
        normalize_config(dict(config) | {"device": device}) for config in configs
    ]
    if not configs:
        raise ValueError("at least one experiment configuration is required")
    identities = [run_identity(config) for config in configs]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate experiment identities")
    plan = resource_plan(
        workers,
        device,
        cpu_budget=cpu_budget,
        threads_per_worker=threads_per_worker,
        gpu_tokens=gpu_tokens,
    )
    # Unused slots do not start processes; keep the allocation record explicit.
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    session = uuid.uuid4().hex
    launch_dir = root / ".launches" / session
    launch_dir.mkdir(parents=True)
    paths = []
    for index, config in enumerate(configs):
        path = launch_dir / f"config-{index:06d}.json"
        path.write_text(json.dumps(config, sort_keys=True) + "\n")
        paths.append(path.resolve())
    started = time.perf_counter()

    def run_slot(slot):
        records = []
        for index in range(slot["worker"], len(configs), len(plan)):
            identity = identities[index]
            directory = root / identity
            directory.mkdir(parents=True, exist_ok=True)
            log = directory / f"worker-{session}.log"
            command = [
                sys.executable,
                "-c",
                _BOOTSTRAP,
                json.dumps(slot["cpus"]),
                str(paths[index]),
                "--run-directory",
                str(directory.resolve()),
            ]
            if resume:
                command.append("--resume")
            before = time.perf_counter()
            with log.open("x", encoding="utf-8") as stream:
                completed = subprocess.run(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env=worker_environment(slot, device),
                    check=False,
                )
            records.append(
                {
                    "run_id": identity,
                    "worker": slot["worker"],
                    "returncode": completed.returncode,
                    "process_wall_seconds": time.perf_counter() - before,
                    "log": str(log),
                }
            )
        return records

    with ThreadPoolExecutor(max_workers=len(plan)) as executor:
        runs = [
            record for records in executor.map(run_slot, plan) for record in records
        ]
    wall_seconds = time.perf_counter() - started
    runs.sort(key=lambda run: identities.index(run["run_id"]))
    scientific = []
    for run in runs:
        path = root / run["run_id"] / "result.json"
        if path.exists() and run["returncode"] == 0:
            result = json.loads(path.read_text())
            run["status"] = result["status"]
            scientific.append(
                {
                    "run_id": run["run_id"],
                    "status": result["status"],
                    "scientific": result["scientific"],
                }
            )
        else:
            run["status"] = "failed"
    completed = sum(run["status"] == "completed" for run in runs)
    report = {
        "schema_version": 1,
        "session": session,
        "device": device,
        "resource_plan": plan,
        "runs": runs,
        "completed": completed,
        "failed": sum(run["status"] == "failed" for run in runs),
        "wall_seconds": wall_seconds,
        "completed_runs_per_second": completed / wall_seconds,
    }
    (launch_dir / "execution.json").write_text(json.dumps(report, indent=2) + "\n")
    (launch_dir / "scientific.json").write_text(json.dumps(scientific, indent=2) + "\n")
    report["report_directory"] = str(launch_dir)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--cpu-budget", type=int)
    parser.add_argument("--threads-per-worker", type=int)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--gpu-devices", help="comma-separated allocated opaque device tokens"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = launch_experiments(
            [json.loads(path.read_text()) for path in args.configs],
            args.output_root,
            workers=args.workers,
            device=args.device,
            cpu_budget=args.cpu_budget,
            threads_per_worker=args.threads_per_worker,
            gpu_tokens=None
            if args.gpu_devices is None
            else args.gpu_devices.split(","),
            resume=args.resume,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, sort_keys=True))
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
