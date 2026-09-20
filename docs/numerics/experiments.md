# Reproducible experiment worker

The installed erot-run command executes one independent experiment. It records
canonical configuration, generated-input digest, seed/RNG state, source-code
digest, available Git revision, runtime versions, device topology and observed
CPU/thread resources. No scheduler or GPU is required for CPU runs.

    erot-run experiments/configs/classical-blocked.json --output-root results
    erot-run experiments/configs/classical-blocked.json --output-root results --resume

The output directory is name plus a configuration digest. For an explicitly
chosen directory, use --run-directory instead. An existing run requires
--resume; incompatible configuration, input, software or topology is rejected.
Output location and --stop-after-chunks do not change scientific run identity.
Use --stop-after-chunks 1 to test a deliberate checkpoint boundary, then resume.
Its result is checkpointed, not completed. Numerical failure returns failed and
a nonzero CLI exit; a deliberate partial stop returns zero with explicit status.

## Configuration schema v1

JSON fields are validated; unknown keys and unsupported controls are rejected.
Default precision is float64 on CPU. Set device to gpu for a CUDA environment;
GPU selection fails rather than silently falling back.

| Fields | Meaning |
|---|---|
| schema_version, name | Version 1 and portable trial label |
| kind, backend | flow: pdhg/sinkhorn/blocked; classical: sinkhorn/blocked; quantum: entropy/quadratic |
| n, m, features | Grid or subsystem sizes; classical point-cloud feature count; m defaults to n |
| seed, dtype, device | Reproducibility and explicit numerical platform |
| epsilon, tolerance | Regularization and solver convergence controls |
| max_iterations | Per-JKO-step budget for flows; total solver update cap otherwise |
| chunk_size | Physical steps per chunk for flows; solver updates per chunk otherwise |
| time_step, steps, energy | Flow horizon and entropy/quadratic energy |
| inner_tolerance, inner_iterations | Entropic flow's inner transport controls |
| block_size | Static tile side for implicit classical/flow transport |
| snapshot_stride | Flow snapshot spacing within each chunk, or null to omit |

The provided flow grid is the one-dimensional unit interval with cell centers
and masses. Entropy uses cell volume 1/n and initial density 1+0.2*cos(pi*x);
quadratic flow uses a seeded positive random mass vector, uniform target and
weight 2. Classical experiments use seeded Gaussian point clouds and positive
random marginals. Quantum experiments use seeded Hermitian costs and full-rank
density matrices. A configuration records all fields, including controls that
apply only to other workload kinds; every field contributes to identity.

## State, output and restart

Each chunk synchronizes device work and publishes the complete state using
[checkpoint format v1](../development/checkpoint-format.md). Flow physical time
advances only after accepted steps. Failed subproblems retain continuation
state; explicit resume can spend another configured per-step budget.
Solver-only runs reaching their total cap without convergence are failed;
changing that scientific configuration requires a new run identity.

Snapshots are separate NPZ files with masses and physical times, including
each chunk's initial sample. They do not replace checkpoints. Only snapshot
files referenced by published checkpoint generations are authoritative; a crash
can leave an unreferenced file. The worker holds an advisory lock for the entire
run and never removes the stable lock filename. The filesystem must implement
the documented POSIX locking/rename/fsync semantics.

result.json separates scientific diagnostics from execution timing. Flow
results include energy, mass, time and accepted steps; solver results include
objective and convergence diagnostics. Nonfinite values are JSON null with
failure status retained. State arrays remain in the checkpoint rather than
being dumped into console JSON.

timing.jsonl records each published chunk's first-specialization compile-and-execute or
repeated-specialization execution time, plus checkpoint time. Worker setup includes backend,
input generation and initial-state work. Parent-process throughput measurements
also include interpreter/import startup. A crash immediately after checkpoint
publication can leave a missing timing event; numerical restart remains valid,
and timing coverage must be checked before throughput claims. No GPU throughput
is established by CPU restart tests.

The initial implementation accepts only the same process/device topology and
runtime/source identity. It stages checkpoint arrays on host; transfer and
host-memory costs are part of capacity planning. Resharded restart is not
advertised. Historical jko_lab/QOTLib checkouts and SDPLab are not dependencies.

## Independent worker launcher

    erot-launch experiments/configs/classical-blocked.json experiments/configs/quantum-entropy.json --output-root results/batch --device cpu --workers 2 --cpu-budget 8

The parent uses only the standard library. It assigns disjoint CPU affinity and
thread environments before fresh child processes import numerical libraries.
On GPU, CUDA_VISIBLE_DEVICES supplies the allocated opaque tokens; by default
one worker uses each token. An explicit --gpu-devices list must be a subset of
that visible allocation. If visibility is absent outside a scheduler, provide
an explicit list of devices you have allocated. No node-wide GPU discovery is
performed. CPU affinity bounds execution resources; BLAS/OpenMP environment
values are not advertised as universal JAX thread-pool controls.

Each launch writes `.launches/<session>/execution.json` and `scientific.json`.
Execution reports contain resource assignment, process exit codes, per-process
wall time, total wall time and completed-run throughput. Scientific values are
separate. Per-run logs have unique session filenames. Duplicate identities in
one launch fail before workers start; concurrent launchers are protected by the
worker's run ownership lock. A failed child cannot be reported successful using
a stale result file. Resume requires --resume and the worker's compatibility
checks still apply. See [Slurm templates and profiles](../../hpc/README.md).

## Throughput validation

Run the same configuration list with a fixed total CPU budget and fresh output
roots. The benchmark refuses an existing root, runs worker-count trials
sequentially, and writes throughput.json after each trial:

    python benchmarks/experiment_throughput.py experiments/configs/classical-blocked.json experiments/configs/quantum-entropy.json --output-root benchmark-results/throughput-cpu --workers 1 2 --cpu-budget 2

For an allocated Hopper-or-newer environment, add --device gpu, set the CPU
budget to the actual allocation and choose worker counts within the visible GPU
count. Use --repeats for variability; compare representative workload lists and
problem sizes before recommending concurrency. A larger throughput can trade
against per-experiment latency. These commands prepare GPU validation and do
not imply it has run.

Each report separates first-specialization compile-and-execute, repeated execution,
checkpoint duration, worker setup and process wall time. Outside-worker-session
time includes interpreter/import startup and exit; it is not a pure import
microbenchmark. Lifetime chunk timing coverage must contain exactly one valid
event per checkpointed chunk, with finite nonnegative durations. Missing,
duplicate, malformed or truncated events mark the measurement incomplete.
Scientific completion remains separate from this timing audit. Measurements
include fresh compilation for each subprocess and checkpoint overhead, so they
measure end-to-end experiment throughput rather than warmed kernel throughput.

Compilation signatures include flow static chunk length and state tree/shape/dtype/
weak-type metadata. A shorter final flow chunk is a new specialization, even
after earlier chunks have run. The combined field includes first-call overhead
and execution; persistent compilation caches can reduce its compilation cost.
