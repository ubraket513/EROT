"""Isolated synchronized distributed solve measurements; no hardware extrapolation."""

import argparse
import hashlib
import json
import os
import time
from contextlib import ExitStack
from pathlib import Path

from erot.runtime.distributed import initialize_runtime


def run(config, output_directory, repeats=3, trace=False):
    import jax
    import numpy as np
    from jax.experimental import multihost_utils as mhu

    from erot.experiments import _atomic_json
    from erot.runtime.config import normalize_config
    from erot.runtime.distributed import make_row_mesh
    from erot.runtime.distributed_checkpoint import collective_call
    from erot.runtime.distributed_inputs import (
        assert_rank_agreement,
        create_inputs,
        local_scalar,
    )
    from erot.runtime.environment import (
        observed_resources,
        runtime_versions,
        source_identity,
    )
    from erot.solvers.distributed_sinkhorn import make_distributed_sinkhorn

    if __package__:
        from .benchmark_blocked_sinkhorn import compiled_memory
        from .benchmark_solvers import _memory_stats
    else:
        from benchmark_blocked_sinkhorn import compiled_memory
        from benchmark_solvers import _memory_stats

    def validate():
        value = normalize_config(config)
        if value["kind"] != "classical" or value["backend"] != "blocked":
            raise ValueError("benchmark requires classical blocked configuration")
        if type(repeats) is not int or repeats < 1:
            raise ValueError("repeats must be a positive integer")
        return value

    config = collective_call("benchmark configuration", validate)
    root = Path(output_directory).resolve()
    assert_rank_agreement(
        {"config": config, "root": str(root), "repeats": repeats, "trace": trace},
        "benchmark controls",
    )
    rank = jax.process_index()
    collective_call(
        "new benchmark directory",
        lambda: root.mkdir(parents=True, exist_ok=False) if rank == 0 else None,
    )
    jax.config.update("jax_enable_x64", config["dtype"] == "float64")
    mesh = make_row_mesh()
    if any(device.platform != config["device"] for device in mesh.devices.flat):
        raise ValueError("benchmark device does not match runtime")
    source = collective_call("benchmark source identity", source_identity)
    runtime = collective_call("benchmark runtime identity", runtime_versions)
    assert_rank_agreement(
        {"source_digest": source["digest"], "runtime": runtime}, "benchmark software"
    )
    inputs = create_inputs(config, mesh)
    solve = make_distributed_sinkhorn(mesh, block_size=config["block_size"])
    arguments = (
        inputs.x,
        inputs.y,
        inputs.a,
        inputs.b,
        config["epsilon"],
        config["tolerance"],
        config["max_iterations"],
        inputs.initial,
    )

    def maximum_duration(started):
        return float(
            np.asarray(
                mhu.process_allgather(np.asarray(time.perf_counter() - started))
            ).max()
        )

    start = time.perf_counter()
    executable = collective_call(
        "compile distributed solve", lambda: solve.lower(*arguments).compile()
    )
    compile_seconds = maximum_duration(start)
    mhu.sync_global_devices("distributed-benchmark-first")
    start = time.perf_counter()
    result = jax.block_until_ready(executable(*arguments))
    first_seconds = maximum_duration(start)
    samples = []
    validity = []
    stack = ExitStack()
    try:
        if trace:
            collective_call(
                "start profiler",
                lambda: stack.enter_context(
                    jax.profiler.trace(str(root / f"profile-rank-{rank:06d}"))
                ),
            )
        for index in range(repeats):
            mhu.sync_global_devices(f"distributed-benchmark-sample-{index}")
            start = time.perf_counter()
            result = jax.block_until_ready(executable(*arguments))
            samples.append(maximum_duration(start))
            diag = result[1]
            validity.append(
                int(local_scalar(diag.status)) == 0
                and float(local_scalar(diag.error)) <= config["tolerance"]
            )
    finally:
        collective_call("stop profiler", stack.close)
    state, diagnostics = result
    valid = bool(
        np.asarray(mhu.process_allgather(np.asarray(all(validity), np.int32))).all()
    )
    error = float(local_scalar(diagnostics.error))
    n, m = inputs.x.shape[0], inputs.y.shape[0]
    case = {
        key: config[key]
        for key in (
            "n",
            "m",
            "features",
            "seed",
            "dtype",
            "epsilon",
            "tolerance",
            "max_iterations",
        )
    }
    case.update(
        generator="numpy-row-seedsequence-v1-uniform-marginals",
        objective="Shannon two-marginal squared-Euclidean OT",
    )
    report = {
        "schema_version": 1,
        "valid": valid,
        "case": case,
        "case_digest": hashlib.sha256(
            json.dumps(case, sort_keys=True).encode()
        ).hexdigest(),
        "block_size": config["block_size"],
        "topology": inputs.topology,
        "iterations": int(local_scalar(state.iterations)),
        "status": int(local_scalar(diagnostics.status)),
        "error": error if np.isfinite(error) else None,
        "compilation_seconds": compile_seconds,
        "first_execution_seconds": first_seconds,
        "warm_seconds": samples,
        "warm_median_seconds": float(np.median(samples)),
        "communication_share": None,
        "profile_captured": trace,
        "memory_interpretation": "compiler buffers and cumulative process allocator counters; not isolated live solve peaks",
    }

    def rank_record():
        record = {
            "rank": rank,
            "source": source,
            "runtime": runtime,
            "resources": observed_resources(),
            "compiler_memory": compiled_memory(executable, n, m),
            "local_compiler_memory": compiled_memory(executable, n // mesh.size, m),
            "device_memory": _memory_stats(jax.local_devices()),
        }
        _atomic_json(root / f"rank-{rank:06d}.json", record)

    collective_call("rank benchmark record", rank_record)

    def publish():
        if rank == 0:
            report["ranks"] = [
                json.loads((root / f"rank-{index:06d}.json").read_text())
                for index in range(jax.process_count())
            ]
            _atomic_json(root / "report.json", report)

    collective_call("benchmark report", publish)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--coordinator")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--process-id", type=int, default=0)
    parser.add_argument(
        "--local-device-ids", help="comma-separated local visible ordinals"
    )
    args = parser.parse_args()
    os.environ["JAX_PLATFORMS"] = "cpu" if args.device == "cpu" else "cuda"
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

        from erot.runtime.config import normalize_config
        from erot.runtime.distributed_checkpoint import collective_call

        config = collective_call(
            "read benchmark configuration",
            lambda: (
                normalize_config(json.loads(args.config.read_text()))
                | {"device": args.device}
            ),
        )
        report = run(config, args.output_directory, args.repeats, args.trace)
        if jax.process_index() == 0:
            print(json.dumps(report, sort_keys=True), flush=True)
        return 0 if report["valid"] else 1
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    finally:
        if initialized:
            import jax

            jax.distributed.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
