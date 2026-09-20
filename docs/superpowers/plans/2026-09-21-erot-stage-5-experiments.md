# Stage 5 reproducible experiments implementation plan

> Execute inline with superpowers:executing-plans under the full-roadmap
> authorization. Keep local commits; no actual scheduler submission is possible.

**Goal:** installed, reproducible independent experiment workers with atomic
checkpoints, exact compatible restart and resource-aware local/Slurm launch.
**Architecture:** portable schema/configuration and checkpoint modules, a
chunked installed worker, and a standard-library parent launcher assigning
disjoint CPU/device resources before child JAX initialization.
**Tech stack:** Python standard library, NumPy, JAX; no mandatory scheduler,
database, optimizer or checkpoint framework.
**Spec:** ../specs/2026-09-21-erot-flows-hpc-design.md, independent experiments
and checkpoint sections; roadmap Stage 5.

## Global constraints

One worker per allocated GPU by default. CPU smoke workers are supported.
No fixed GPU model/count/core limit, account, partition, QoS or personal path.
Use available affinity and explicit/scheduler limits; reject oversubscription.
Do not infer GPU throughput from CPU workers. Preserve source checkouts/SDPLab.
Serialization must not require pickle or dynamic untrusted imports.
Only same-topology restart is initially supported.

## Review focus

Interrupted publication must leave the previous complete checkpoint usable.
Changed configuration, input identity, dtype or topology must fail restart.
Concurrent duplicate experiment identities must not overwrite one another.
Worker thread environment and disjoint CPU affinity must precede backend use.
Failed numerical steps must retain complete state and never claim a completed run.

## Task 1: atomic versioned checkpoints

Files: src/erot/runtime/{__init__,checkpoint}.py and checkpoint unit tests.
save_checkpoint(path,state,metadata) creates a new immutable generation and
publishes LATEST only after data/manifest are flushed. load_checkpoint(path,
expected_metadata) verifies schema, hashes, array shapes/dtypes and compatible
metadata before reconstructing whitelisted named tuple states.

- [x] Write round-trip tests for FlowState (both backends), Sinkhorn and quantum
  states, including complex arrays and integer counters.
- [x] Write incompatible-config/topology/dtype and truncated/corrupt/incomplete
  generation tests. Initial run must fail on missing module.
- [x] Encode array leaves in non-pickle npz and tree structure in JSON with
  explicit known-state registry. Include schema, checksums and required identity
  metadata. Use temporary generation, fsync, rename, then atomic latest pointer.
- [x] Simulate interruption before pointer publication; the old checkpoint must
  remain loadable. Reject unknown schemas/types and invalid hashes.
- [x] Run focused tests, expected pass; commit.

## Task 2: run identity and installed chunked worker

Files: src/erot/experiments.py, src/erot/runtime/environment.py,
experiments/configs/*.json, experiments/run.py, tests/integration/.
Configuration schema v1 supports flow studies (heat entropy or quadratic,
PDHG or dense/blocked Sinkhorn) and solver-only classical/quantum entropy
experiments with bounded solver chunks. Explicit seed, sizes, epsilon, time
step, tolerances, dtype, output policy and total work/steps are recorded.
A canonical scientific configuration digest plus trial label identifies a run;
output location and deliberate stop-after-chunks control do not change identity.

- [x] Add uninterrupted versus one-chunk/checkpoint/resume tests, comparing full
  state and final scientific output. Add changed-config rejection and numerical
  failure records. Run before implementation, expected missing worker.
- [x] Generate inputs deterministically; record input/geometry digest, seed and
  RNG state, source-code digest/revision when available, runtime versions and
  selected device/topology. Use a per-run ownership lock to prevent collisions.
- [x] Run finite chunks, synchronize before saving, checkpoint complete solver
  state and physical time. Save requested snapshots separately. On failed solve,
  persist state and explicit reason; do not advance accepted physical time.
- [x] Provide installed erot-run entry point and repository wrapper. Support
  resume and deliberate stop-after-chunks for restart validation.
- [x] Run CPU examples and installed worker smoke, expected reproducible;
  commit.

## Task 3: resource-aware independent workers

Files: src/erot/runtime/environment.py, experiments/launch.py or installed
launcher subcommand, hpc/run_array.sbatch, hpc/profiles/*.json, launcher tests.
Parent must avoid initializing JAX devices. Assign one visible GPU token per
worker and disjoint CPU affinity subsets; all resource choices are recorded.

- [x] Test one/two CPU workers produce independent identities/results, and reject
  more workers/threads/devices than allocation. Test resource-plan construction
  with supplied GPU UUID tokens without needing hardware.
- [x] Derive affinity from os.sched_getaffinity and respect scheduler/explicit
  budgets. Set OMP/OpenBLAS/MKL thread limits and worker affinity before numerical
  backend use. Record actual observed affinity, not only requested counts.
- [x] Launch workers using fresh subprocesses; isolate outputs/logs by run id.
  Preserve full failure exit status. Add aggregation of scientific output
  separately from execution timing.
- [x] Add configurable Slurm array template and profile examples without site
  account/model constraints. Site sbatch resource flags remain external inputs.
- [x] Run one/two-worker CPU smoke and collision/restart tests; commit.

## Task 4: throughput tools, documentation and review

- [x] Record startup, compilation/first chunk, subsequent chunks and checkpoint
  overhead separately in each run. Measure one/two CPU worker throughput in
  isolated runs; prepare equivalent GPU commands with unmeasured labels.
- [x] Document schema, supported workloads, exact-restart compatibility,
  checkpoint ownership, resource assumptions and HPC launch commands.
- [x] Run full CPU suite, Ruff, package/build/install and installed worker
  round-trip. Request one final fresh review; fix material findings with
  regression tests and a green suite.
- [x] Commit verification evidence and continue Stage 6 distributed solves.
