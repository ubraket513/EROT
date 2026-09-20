# Stage 6 distributed individual solves implementation plan

> Execute inline with superpowers:executing-plans under the existing full-roadmap
> authorization. One fresh final review; no implementation agents. Hardware
> validation is prepared, not invented, because no GPU/Slurm allocation exists.

**Goal:** an experimental row-partitioned implicit classical Sinkhorn solver,
complete same-topology distributed restart and reproducible validation tools.
**Architecture:** one global row mesh, local source points/masses/potentials,
replicated target points/masses/potentials, bounded local geometry tiles and
explicit stable collectives. A host runtime initializes before device access,
constructs global arrays from local data and owns distributed publication.
**Tech stack:** JAX shard_map and named sharding, NumPy host inputs, standard
library configuration/publication; no mandatory MPI/native dependency.
**Spec:** ../specs/2026-09-21-erot-flows-hpc-design.md, distributed solve,
checkpoint and support matrix sections; roadmap Stage 6.

## Global constraints

Preserve base Python>=3.11/JAX>=0.4.30 until Stage8 floor verification. New
experimental distributed code is tested separately against Python3.12.13 and
JAX/jaxlib0.11.2. Record a reproducible pinned CPU validation environment before
adding version-sensitive code; runtime must reject unsupported distributed API
capability explicitly. Do not advertise older versions for this module.
No dense cost/coupling construction inside the distributed solver. All-masked
local support must contribute zero to global sums, never NaN. Every participant
executes the same collective sequence. No source checkout or SDPLab changes.

Actual two-GPU/two-process and interconnect/capacity gates remain unexecuted.
Virtual-device and local multiprocess CPU checks are correctness evidence only.
No quantum sharding claim: full PSD projection and entropy Gibbs recovery remain
dense spectral operations. Dense PDHG must be evaluated separately.

## Review focus

A rank with zero source support must participate without producing NaNs.
A rank-local invalid input must terminate consistently across the mesh.
Uneven padded partitions must not create transport mass or bias the gauge.
Checkpoint publication interrupted before all ranks finish must preserve the
previous globally complete generation.
Global arrays must be constructed from local data; no host gathering of full
source geometry, potentials or coupling in the production path.

## Task 1: distributed runtime and reproducible local inputs

Files: src/erot/runtime/distributed.py, requirements/distributed-cpu.txt,
tests/distributed/runtime_probe.py, tests/integration/test_distributed_runtime.py,
docs/numerics/distributed.md.

Interfaces:
- initialize_runtime(coordinator_address=None,num_processes=1,process_id=0,
  local_device_ids=None): initialize JAX before querying devices for multiprocess;
  explicit process arguments validated and not inferred from arbitrary env.
- make_row_mesh(): one-dimensional named mesh 'rows' of all participating devices.
- global_from_local(local_array,mesh,replicated=False): NamedSharding P('rows')
  or P(), using make_array_from_process_local_data; host local shape explicit.
- partition_rows(n,parts,index): equal padded extent ceil(n/parts), global start,
  valid row count. Source generation consumes only local global row indices.

- [ ] Write missing-module tests for deterministic uneven partitions and fresh
  subprocess initialization with two forced host devices.
- [ ] Record installed dependency versions and create the pinned CPU environment.
  Use current official JAX documentation for shard_map and process-local arrays.
- [ ] Implement initialization, mesh and array construction. Reject late backend
  initialization with the runtime's explicit error, rather than silently falling
  back to single-process operation.
- [ ] Probe one process/two CPU devices: global shape, addressable shards,
  replicated target data and partition metadata agree with requested inputs.
- [ ] Run focused tests, record actual outputs, commit.

## Task 2: collective implicit Sinkhorn

Files: src/erot/solvers/distributed_sinkhorn.py,
tests/distributed/solver_probe.py, tests/integration/test_distributed_sinkhorn.py.
Consume PointCloudGeometry, streamed_logsumexp, SinkhornState/SolverDiagnostics,
_checked_count and Task1 mesh. Produce:
make_distributed_sinkhorn(mesh,block_size=128): returns a compiled function
solve(x,y,a,b,epsilon,tolerance,max_iterations,state) -> (state,diagnostics).
Global x/a/f have a padded row dimension sharded on 'rows'; y/b/g and iteration
counter are replicated. Caller supplies initial SinkhornState. This first
interface supports squared Euclidean point clouds, two marginals and explicit
same-problem state only; existing general dense and warm-start APIs remain.

- [ ] Write a fresh-process two-device test comparing to dense and blocked solves
  for n=7,m=5 (padded to8), complex inputs rejected, both float32/64, small epsilon,
  all support concentrated on one shard and zero target entries. Initially fail
  on missing module. Check objective through implicit evaluation, potentials in
  weighted gauge, global residual and exact restart counters.
- [ ] Implement local row update using complete replicated targets. For column
  local log-sums l_r, use M=pmax(l_r), S=psum(exp(l_r-M)) with safe M=0 when all
  entries are -inf, then L=M+log(S); all-empty output remains -inf.
- [ ] Compute row L1 residual with psum of local errors, column L1 residual from
  globally reduced log-sums, and take their maximum. Gauge shift uses psum of
  local weighted f and mass; apply f-=shift,g+=shift on positive supports only.
- [ ] Reduce validity across all ranks before a globally uniform while_loop;
  iteration count, failure/status and residual outputs are replicated. Check
  numerical controls, finite geometry, nonnegative matched masses and counters.
- [ ] Test one rank's invalid marginal/control input yields a consistent failure,
  budget exhaustion and zero-budget resume preserve state, and split budgets
  reproduce an uninterrupted solve. Inspect compiled buffers for bounded tiles.
- [ ] Record per-sweep collective contract (target-vector max/sum and scalar
  residual/gauge/validity reductions), run focused references, commit.

## Task 3: multiprocess execution and distributed checkpoints

Files: src/erot/runtime/distributed_checkpoint.py,
src/erot/distributed.py, hpc/run_distributed.sbatch,
tests/distributed/process_probe.py, tests/integration/test_distributed_process.py.
Consume Task1 initialization/local arrays and Task2 solve. Reuse checkpoint
encoding for local complete states; do not serialize a non-addressable global
array via NumPy. Each process stores its addressable f shards, replicated g,
iteration counters, local input identity and global partition metadata.

- [ ] Write a two-process localhost CPU test with explicit coordinator address,
  process IDs and one device each. Launch together with bounded timeout, capture
  separate logs, and verify consistent residual/status/iterations against the
  one-process reference. No scheduler needed for this correctness test.
- [ ] Implement a bounded chunk CLI with deterministic local input generation
  using global row indices, explicit mesh topology and shared configuration.
  All ranks must execute solve/checkpoint collectives, rank0 prints summaries.
- [ ] Add save_distributed_checkpoint/load_distributed_checkpoint: each rank
  writes an immutable local generation; only after every rank confirms durable
  success does rank0 atomically publish a global manifest listing all exact
  generations. Never select each rank's latest independently on restart.
- [ ] Test same-topology stop/resume, missing/corrupt rank data, changed topology,
  source/input/dtype rejection and simulated incomplete global publication.
  Keep global ownership on rank0 and coordinate errors before collective phases.
- [ ] Add site-configurable Slurm launch template with explicit coordinator,
  world size and rank mapping, before device access. No site/account/GPU model
  constants. Preserve scheduler GPU visibility; document shared filesystem needs.
- [ ] Run local two-process tests, package CLI smoke, commit.

## Task 4: scaling tools, limits and independent review

Files: benchmarks/distributed_sinkhorn.py, tests/gpu/test_distributed_gpu.py,
docs/performance/distributed-transport.md, docs/development/stage-6-validation.md.

- [ ] Add isolated one/two-device matched-tolerance measurements with warm-up,
  synchronized solve time, local/global shapes, process/device topology, compiler
  memory and available per-device runtime memory. Mark unavailable counters null.
- [ ] Prepare one-process/two-GPU and two-process/one-GPU validation commands and
  opt-in tests; report numerical equivalence before speed/capacity comparisons.
  Profiler capture instructions separate collective time from computation; do
  not invent a communication share from static byte estimates.
- [ ] Evaluate dense PDHG state storage and column-residual communication as a
  separate documented path; do not advertise implementation/scaling without its
  own tests. Document dense quantum spectral capability and unmeasured GPU limit.
- [ ] Document virtual CPU correctness results separately from unexecuted actual
  GPU peak memory, speedup, capacity and interconnect gates. Multi-node promotion
  stays gated. No1.3x speedup claim without measurement.
- [ ] Run full CPU checks, package/installed workflow, focused multiprocess checks
  and one fresh final review; fix material findings with RED/GREEN regressions.
- [ ] Commit validation and continue Stage7 native decision based on evidence.

## Plan self-review

Row ownership, stable reductions and rank-uniform termination are covered by
Task2; local inputs and pre-device initialization by Task1/3; complete distributed
publication by Task3; hardware tools/limits and separate PDHG/quantum assessment
by Task4. The user's latest hardware instruction replaces execution of absent
GPU gates with prepared tools and explicit unmeasured status, not a pass claim.
This plan does not add quantum gradient flows or distributed spectral solvers.
