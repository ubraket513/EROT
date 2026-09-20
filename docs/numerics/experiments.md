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

timing.jsonl records each published chunk's first-call compile-and-execute or
subsequent execution time, plus checkpoint time. Worker setup includes backend,
input generation and initial-state work. Parent-process throughput measurements
also include interpreter/import startup. A crash immediately after checkpoint
publication can leave a missing timing event; numerical restart remains valid,
and timing coverage must be checked before throughput claims. No GPU throughput
is established by CPU restart tests.

The initial implementation accepts only the same process/device topology and
runtime/source identity. It stages checkpoint arrays on host; transfer and
host-memory costs are part of capacity planning. Resharded restart is not
advertised. Historical jko_lab/QOTLib checkouts and SDPLab are not dependencies.
