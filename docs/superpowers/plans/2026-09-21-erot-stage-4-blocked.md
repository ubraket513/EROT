# Stage 4 blocked classical transport implementation plan

> Use superpowers:executing-plans inline under the user's full-roadmap authorization.

**Goal:** solve two-marginal point-cloud Shannon OT without storing n-by-m cost
or coupling arrays, retaining explicit plan blocks and transport application.
**Architecture:** PyTree geometry objects provide cost tiles; streamed reductions
feed a separate blocked Sinkhorn implementation sharing state/status conventions.
The existing dense multi-marginal solver remains available. No grid/convolution
path is justified by current evidence and none is implied by this milestone.
**Tech stack:** JAX/NumPy; CPU compiled-memory evidence and configurable GPU tools.
**Spec:** ../specs/2026-09-21-erot-flows-hpc-design.md and roadmap Stage 4.

## Global constraints

Match dense Shannon objective epsilon*sum(pi*(log(pi)-1)), cost-unit potentials
and weighted gauge. Preserve zero supports, arbitrary positive common mass,
additional-work resume and distinct warm start. No dense point-cloud output by
default, no implicit host conversion or I/O. Block size is static compilation
configuration. No GPU measurements without hardware; prepare tools instead.

## Review focus

Nondivisible tiles and empty-support tiles must contribute no padding mass.
Very small epsilon must use log reductions without raw-kernel underflow.
Point-coordinate conditioning must not silently change squared-distance costs.
Resume with zero budget and wide counters must preserve valid counts/status.
Transport application must match explicit plan multiplication for signed vectors.

## Task 1: geometry and stable tiled reductions

Files: src/erot/geometry/{__init__,dense,pointcloud,reductions}.py and unit tests.
DenseGeometry(cost) and PointCloudGeometry(x,y) are PyTrees. Point-cloud cost is
squared Euclidean distance; x and y may have unequal counts but equal feature
dimension. Tile access takes static block size and block indices, pads safely
and masks positions beyond original shapes. Dense geometry is a compatibility
adapter, not a memory-scaling claim.

- [x] Add independent NumPy tile/reference tests, rectangular nondivisible shapes,
  coincident points and explicit invalid shapes/dtypes.
- [x] Run tests before implementation; expect missing-module failure.
- [x] Implement tile evaluation without forming full n*m arrays. Prefer direct
  tile differences when needed to avoid norm-expansion cancellation; account for
  tile feature dimension in memory documentation.
- [x] Implement streamed row/column log-sum-exp for (f_i+g_j-C_ij)/epsilon;
  stable logaddexp accumulation handles all-masked blocks as -inf.
- [x] Verify against dense reductions for zero supports and small epsilon;
  test JIT and batching. Expected all pass; commit.

## Task 2: blocked solve and explicit outputs

Files: src/erot/solvers/blocked_sinkhorn.py, src/erot/geometry/plan.py,
tests/reference/test_blocked_sinkhorn.py and state tests.
Consumes Task1 geometry/reductions. Returns SinkhornState and SolverDiagnostics.
Provide solve_blocked_sinkhorn(geometry,marginals,epsilon,tolerance,max_iterations,
*,block_size=128,state=None,warm_start=None), plan_block and apply_transport.

- [x] Add dense/blocked matched-tolerance tests with nonunit mass, nondivisible
  tiles, inactive supports, small epsilon, exact continuation and invalid input.
  Tests initially fail on missing solver.
- [x] Implement alternating cost-unit potential updates using streamed reductions,
  weighted gauge, streamed marginal L1 residual and finite failure status.
  Preserve checked int32 budgets and zero-work behavior.
- [x] Compute objective through streamed plan tiles, using xlogy at zero; expose
  explicit block retrieval and transport application to vectors/matrices.
- [x] Compare output products/objectives to independent explicit dense plans;
  no default materialization. Run focused suite, expected pass; commit.

## Task 3: memory evidence and benchmark tools

Files: benchmarks/benchmark_blocked_sinkhorn.py, docs/performance/blocked-transport.md,
tests/integration/test_blocked_memory.py and optional GPU checks.
Benchmark separates compile/first-call/warm samples, synchronizes output and
records geometry, block size, dtype, tolerance, objective, residual and device.

- [x] Inspect lowered/compiled computation for point-cloud full n*m intermediates.
  Use compiled memory analysis when supported and record unavailable metrics
  explicitly. CPU evidence must include two sizes with fixed tile size.
- [x] Add reproducible dense comparison for problems fitting memory, fixed-tile
  capacity study and configurable GPU command. Do not infer measured GPU peaks
  from compiler estimates or CPU RSS.
- [x] Document O((n+m)*d+n+m+tile^2*d) storage and quadratic arithmetic; include
  compilation workspace/runtime limitations. Link output policy and numerical
  conditioning. Run CPU benchmark; expected matched numerical results.

## Task 4: blocked transport in entropic gradient flows

Files: src/erot/flows/{jko,trajectory,_precision}.py and flow integration tests.
Accept geometry objects through the existing cost position for the Sinkhorn
backend, with static transport_block_size. Array inputs retain dense behavior;
PDHG continues to require an explicit dense cost and coupling.

- [x] Compare two accepted blocked-geometry JKO steps to dense steps at the same
  epsilon, objective and tolerance, under JIT and nondivisible tiles. Before
  implementation expect geometry-to-array conversion failure.
- [x] Route geometry inputs through blocked Sinkhorn and streamed primal
  objective/total transport mass. Preserve f/(2*dt), dual descent, complete
  resume and physical-time failure semantics. Promote geometry leaves together
  with state; do not create a hidden dense cost in trajectories.
- [x] Inspect a compiled entropic chunk for full coupling/cost shapes, test
  inner failure keeps physical time unchanged, and reject geometry PDHG clearly.
- [x] Run focused tests; expected pass. Document and commit.

## Task 5: integration verification and review

- [x] Run full CPU tests and Ruff, build/install outside checkout and smoke a
  point-cloud solve and transport application.
- [x] Request final fresh review and fix material findings through regressions.
- [x] Record actual memory evidence and outstanding real-GPU measurements;
  mark completed software gates accurately and proceed to cluster experiments.
