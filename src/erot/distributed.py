"""Run one experimental distributed classical solve with complete rank restart."""

import argparse
import json
import os
import time
from contextlib import ExitStack
from pathlib import Path

from .runtime.config import config_digest, normalize_config
from .runtime.distributed import initialize_runtime
from .runtime.environment import (
    observed_resources,
    run_lock,
    runtime_versions,
    source_identity,
)


def run_distributed(config, run_directory, *, resume=False, stop_after_chunks=None):
    """All initialized processes cooperate; rank0 owns the shared run directory."""
    import jax
    import numpy as np
    from jax.experimental import multihost_utils as mhu

    from .experiments import _atomic_json
    from .runtime.distributed import make_row_mesh
    from .runtime.distributed_checkpoint import (
        collective_call,
        load_distributed_checkpoint,
        save_distributed_checkpoint,
    )
    from .runtime.distributed_inputs import (
        assert_rank_agreement,
        create_inputs,
        local_scalar,
        local_state,
        restore_state,
    )
    from .solvers.distributed_sinkhorn import make_distributed_sinkhorn
    from .solvers.state import SolverDiagnostics

    started = time.perf_counter()
    rank = jax.process_index()

    def validate():
        value = normalize_config(config)
        if value["kind"] != "classical" or value["backend"] != "blocked":
            raise ValueError(
                "distributed driver requires classical blocked configuration"
            )
        if stop_after_chunks is not None and (
            type(stop_after_chunks) is not int or stop_after_chunks < 1
        ):
            raise ValueError("stop_after_chunks must be a positive integer")
        return value

    config = collective_call("distributed configuration", validate)
    root = Path(run_directory).resolve()
    assert_rank_agreement(
        {
            "config": config,
            "root": str(root),
            "resume": resume,
            "stop_after_chunks": stop_after_chunks,
        },
        "configuration and run controls",
    )
    jax.config.update("jax_enable_x64", config["dtype"] == "float64")
    mesh = make_row_mesh()
    expected_platform = "cpu" if config["device"] == "cpu" else "gpu"
    if any(device.platform != expected_platform for device in mesh.devices.flat):
        raise ValueError("requested device platform does not match initialized runtime")
    source = collective_call("source identity", source_identity)
    software = collective_call(
        "runtime identity",
        lambda: {
            "runtime": runtime_versions(),
            "source_digest": source["digest"],
        },
    )
    assert_rank_agreement(software, "source and runtime")
    with ExitStack() as stack:
        collective_call(
            "global run ownership",
            lambda: stack.enter_context(run_lock(root)) if rank == 0 else None,
        )
        inputs = create_inputs(config, mesh)
        metadata = {
            "config_digest": config_digest(config),
            "input_digest": inputs.input_digest,
            "dtype": config["dtype"],
            "topology": inputs.topology,
            "seed": config["seed"],
            "generator": "numpy-row-seedsequence-v1-uniform-marginals",
            **software,
        }
        local_metadata = metadata | inputs.local_metadata
        record = {
            "schema_version": 1,
            "config": config,
            "checkpoint_metadata": metadata,
            "source_revision": source["revision"],
        }

        def record_run():
            path = root / "run.json"
            if path.exists():
                if not resume:
                    raise ValueError("existing distributed run requires --resume")
                previous = json.loads(path.read_text())
                if not isinstance(previous, dict) or any(
                    previous.get(key) != record[key]
                    for key in ("schema_version", "config", "checkpoint_metadata")
                ):
                    raise ValueError(
                        "incompatible distributed run configuration, inputs, runtime or topology"
                    )
            else:
                _atomic_json(path, record)

        collective_call(
            "distributed run record", record_run if rank == 0 else lambda: None
        )
        checkpoints = root / "checkpoints"
        available = (checkpoints / "LATEST").exists()
        assert_rank_agreement(available, "checkpoint visibility")
        if resume and available:
            payload, _ = load_distributed_checkpoint(checkpoints, local_metadata)
            state = restore_state(payload["state"], mesh)
            chunks, last_status = payload["chunks"], int(payload["diagnostics"].status)
            error = float(payload["diagnostics"].error)
        else:
            state, chunks, last_status, error = inputs.initial, 0, 1, None
        assert_rank_agreement(
            {
                "iterations": int(local_scalar(state.iterations)),
                "chunks": chunks,
                "status": last_status,
            },
            "restored state progress",
        )
        solve = make_distributed_sinkhorn(mesh, block_size=config["block_size"])
        setup_seconds = float(
            np.asarray(
                mhu.process_allgather(np.asarray(time.perf_counter() - started))
            ).max()
        )
        session_chunks = 0
        status = "checkpointed"
        while (
            last_status != 0
            and int(local_scalar(state.iterations)) < config["max_iterations"]
        ):
            remaining = config["max_iterations"] - int(local_scalar(state.iterations))
            before = time.perf_counter()
            state, diagnostics = solve(
                inputs.x,
                inputs.y,
                inputs.a,
                inputs.b,
                config["epsilon"],
                config["tolerance"],
                min(config["chunk_size"], remaining),
                state,
            )
            jax.block_until_ready((state, diagnostics))
            duration = float(
                np.asarray(
                    mhu.process_allgather(np.asarray(time.perf_counter() - before))
                ).max()
            )
            chunks += 1
            session_chunks += 1
            last_status = int(local_scalar(diagnostics.status))
            error = float(local_scalar(diagnostics.error))
            payload = collective_call(
                "local checkpoint staging",
                lambda: {
                    "state": local_state(state),
                    "chunks": chunks,
                    "diagnostics": SolverDiagnostics(
                        *(local_scalar(value) for value in diagnostics)
                    ),
                },
            )
            before = time.perf_counter()
            generation = save_distributed_checkpoint(
                checkpoints, payload, local_metadata
            )
            checkpoint_seconds = float(
                np.asarray(
                    mhu.process_allgather(np.asarray(time.perf_counter() - before))
                ).max()
            )
            event = {
                "chunk": chunks,
                "compile_and_execute_seconds"
                if session_chunks == 1
                else "execute_seconds": duration,
                "checkpoint_seconds": checkpoint_seconds,
                "generation": generation.name,
                "status": last_status,
            }

            def append_timing():
                if rank == 0:
                    with (root / "timing.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(event, allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())

            collective_call("distributed timing record", append_timing)
            if last_status not in (0, 1):
                break
            if stop_after_chunks is not None and session_chunks >= stop_after_chunks:
                break
        iterations = int(local_scalar(state.iterations))
        if last_status == 0:
            status = "completed"
        elif last_status not in (0, 1) or iterations >= config["max_iterations"]:
            status = "failed"
        report = {
            "schema_version": 1,
            "status": status,
            "solver_status": last_status,
            "iterations": iterations,
            "error": error if error is None or np.isfinite(error) else None,
            "chunks": chunks,
            "session_chunks": session_chunks,
            "topology": inputs.topology,
            "worker_setup_seconds": setup_seconds,
            "session_wall_seconds": time.perf_counter() - started,
        }
        collective_call(
            "rank resources",
            lambda: _atomic_json(
                root / f"resources-rank-{rank:06d}.json", observed_resources()
            ),
        )
        collective_call(
            "distributed result",
            lambda: _atomic_json(root / "result.json", report) if rank == 0 else None,
        )
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--coordinator")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--process-id", type=int, default=0)
    parser.add_argument(
        "--local-device-ids", help="comma-separated local visible ordinals"
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-chunks", type=int)
    args = parser.parse_args(argv)
    os.environ["JAX_PLATFORMS"] = "cpu" if args.device == "cpu" else "cuda"
    # Affinity is assigned by the site launcher (e.g. srun --cpu-bind=cores).
    threads = min(
        len(os.sched_getaffinity(0)),
        int(os.environ.get("SLURM_CPUS_PER_TASK", len(os.sched_getaffinity(0)))),
    )
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = str(threads)
    initialized = False
    try:
        initialize_runtime(
            args.coordinator,
            args.processes,
            args.process_id,
            None
            if args.local_device_ids is None
            else [int(value) for value in args.local_device_ids.split(",")],
        )
        initialized = args.processes > 1
        import jax

        from .runtime.distributed_checkpoint import collective_call

        config = collective_call(
            "read configuration",
            lambda: (
                normalize_config(json.loads(args.config.read_text()))
                | {"device": args.device}
            ),
        )
        report = run_distributed(
            config,
            args.run_directory,
            resume=args.resume,
            stop_after_chunks=args.stop_after_chunks,
        )
        if jax.process_index() == 0:
            print(json.dumps(report, sort_keys=True), flush=True)
        return 1 if report["status"] == "failed" else 0
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    finally:
        if initialized:
            import jax

            jax.distributed.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
