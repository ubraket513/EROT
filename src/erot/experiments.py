"""Installed deterministic experiment worker with complete-state restart."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path

import jax
import numpy as np

from .runtime.checkpoint import _fsync_directory, load_checkpoint, save_checkpoint
from .runtime.config import config_digest, normalize_config, run_identity
from .runtime.environment import (
    observed_resources,
    run_lock,
    runtime_versions,
    source_identity,
)
from .runtime.workloads import create_workload


def _atomic_json(path, value):
    temporary = path.with_name("." + path.name + "-" + uuid.uuid4().hex)
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _snapshots(path, values, times):
    temporary = path.with_name("." + path.name + "-" + uuid.uuid4().hex)
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("xb") as stream:
        np.savez(stream, values=np.asarray(values), times=np.asarray(times))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _finite(value):
    value = float(value)
    return value if np.isfinite(value) else None


def run_experiment(
    config: dict,
    run_directory: str | Path,
    *,
    resume=False,
    stop_after_chunks: int | None = None,
) -> dict:
    """Run bounded chunks, publishing complete state before reporting progress.

    A deliberate stop_after_chunks returns checkpointed rather than completed.
    A numerical failure is checkpointed and returned as failed; explicit resume
    may spend another configured chunk budget on the same failed subproblem.
    """
    start = time.perf_counter()
    config = normalize_config(config)
    if stop_after_chunks is not None and (
        isinstance(stop_after_chunks, bool)
        or not isinstance(stop_after_chunks, int)
        or stop_after_chunks <= 0
    ):
        raise ValueError("stop_after_chunks must be a positive integer")
    root = Path(run_directory)
    with run_lock(root):
        record_path = root / "run.json"
        previous_record = (
            json.loads(record_path.read_text()) if record_path.exists() else None
        )
        if previous_record is not None:
            if previous_record["config"] != config:
                raise ValueError("incompatible experiment configuration")
            if not resume:
                raise ValueError("run already exists; explicitly select resume")
        elif resume:
            raise ValueError("cannot resume a run without its identity record")
        jax.config.update("jax_enable_x64", config["dtype"] == "float64")
        device = jax.devices(config["device"])[0]
        workload = create_workload(config, device)
        source = source_identity()
        identity = run_identity(config)
        metadata = {
            "config_digest": config_digest(config),
            "input_digest": workload.input_digest,
            "dtype": config["dtype"],
            "topology": {
                "processes": 1,
                "devices": 1,
                "platform": device.platform,
                "device_kind": device.device_kind,
            },
            "runtime": runtime_versions(),
            "source_digest": source["digest"],
            "seed": config["seed"],
            "rng_state": workload.rng_state,
        }
        record = {
            "schema_version": 1,
            "run_id": identity,
            "config": config,
            "checkpoint_metadata": metadata,
            "source_revision": source["revision"],
            "resources": observed_resources(),
        }
        if previous_record is None:
            _atomic_json(record_path, record)
        elif previous_record["checkpoint_metadata"] != metadata:
            raise ValueError(
                "incompatible checkpoint runtime, source or input identity"
            )
        checkpoint_path = root / "checkpoints"
        if resume and (checkpoint_path / "LATEST").exists():
            saved, _ = load_checkpoint(checkpoint_path, metadata)
            state = jax.tree.map(
                lambda x: jax.device_put(x, device) if isinstance(x, np.ndarray) else x,
                saved["state"],
            )
            chunks, last_status = saved["chunks"], saved["last_status"]
        else:
            state, chunks, last_status = workload.initial, 0, 1
        setup_seconds = time.perf_counter() - start
        session_chunks = 0
        last_diagnostics = None
        status = "checkpointed"
        while True:
            completed = (
                int(state.accepted_steps)
                if config["kind"] == "flow"
                else int(state.iterations)
            )
            target = (
                config["steps"]
                if config["kind"] == "flow"
                else config["max_iterations"]
            )
            if config["kind"] != "flow" and last_status == 0:
                status = "completed"
                break
            if completed >= target:
                status = "completed" if config["kind"] == "flow" else "failed"
                break
            count = min(config["chunk_size"], target - completed)
            before = time.perf_counter()
            state, diagnostics, values, times = jax.block_until_ready(
                workload.chunk(state, count)
            )
            chunk_seconds = time.perf_counter() - before
            chunks += 1
            session_chunks += 1
            last_status = int(diagnostics.status)
            last_diagnostics = diagnostics
            snapshot_file = None
            if values.size:
                snapshot_file = f"snapshots/chunk-{chunks:08d}.npz"
                _snapshots(root / snapshot_file, values, times)
            before = time.perf_counter()
            generation = save_checkpoint(
                checkpoint_path,
                {
                    "state": state,
                    "chunks": chunks,
                    "last_status": last_status,
                    "diagnostics": diagnostics,
                    "snapshot_file": snapshot_file,
                },
                metadata,
            )
            checkpoint_seconds = time.perf_counter() - before
            event = {
                "chunk": chunks,
                "first_chunk_in_process": session_chunks == 1,
                "compile_and_execute_seconds"
                if session_chunks == 1
                else "execute_seconds": chunk_seconds,
                "checkpoint_seconds": checkpoint_seconds,
                "generation": generation.name,
                "status": last_status,
            }
            with (root / "timing.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            failed = (
                last_status != 0
                if config["kind"] == "flow"
                else last_status not in (0, 1)
            )
            if failed:
                status = "failed"
                break
            completed = (
                int(state.accepted_steps)
                if config["kind"] == "flow"
                else int(state.iterations)
            )
            if (config["kind"] == "flow" and completed >= target) or (
                config["kind"] != "flow" and last_status == 0
            ):
                status = "completed"
                break
            if config["kind"] != "flow" and completed >= target:
                status = "failed"
                break
            if stop_after_chunks is not None and session_chunks >= stop_after_chunks:
                break
        if (
            last_diagnostics is None
            and resume
            and (checkpoint_path / "LATEST").exists()
        ):
            saved, _ = load_checkpoint(checkpoint_path, metadata)
            last_diagnostics = saved["diagnostics"]
        scientific = {
            key: _finite(value) for key, value in workload.scientific(state).items()
        }
        if config["kind"] == "flow":
            scientific.update(
                {
                    "physical_time": float(state.time),
                    "accepted_steps": int(state.accepted_steps),
                    "mass": _finite(np.asarray(state.rho).sum()),
                    "work": int(state.work),
                }
            )
        if last_diagnostics is not None:
            scientific["error"] = _finite(last_diagnostics.error)
            for name in ("objective", "primal", "dual", "feasibility"):
                if hasattr(last_diagnostics, name):
                    scientific[name] = _finite(getattr(last_diagnostics, name))
        report = {
            "schema_version": 1,
            "run_id": identity,
            "status": status,
            "solver_status": last_status,
            "chunks": chunks,
            "session_chunks": session_chunks,
            "scientific": scientific,
            "worker_setup_seconds": setup_seconds,
            "session_wall_seconds": time.perf_counter() - start,
            "checkpoint": str(checkpoint_path / "LATEST"),
            "resources": observed_resources(),
        }
        _atomic_json(root / "result.json", report)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output-root", type=Path)
    output.add_argument("--run-directory", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-chunks", type=int)
    args = parser.parse_args(argv)
    try:
        config = normalize_config(json.loads(args.config.read_text()))
        directory = args.run_directory or args.output_root / run_identity(config)
        report = run_experiment(
            config,
            directory,
            resume=args.resume,
            stop_after_chunks=args.stop_after_chunks,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
