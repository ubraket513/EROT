# EROT integration and HPC delivery roadmap

License update (2026-09-21): the user confirmed Apache-2.0 for both QOTLib and numerical-gradient-flows. Earlier missing-license observations below describe the inspected trees, not an outstanding permission question. Preserve attribution and applicable notices when adapting source.

Status: Stage 0 completed and independently reviewed; Stage 1 is in progress. See [Stage 0 evidence](../../development/stage-0-validation.md).

**Goal:** first clean and restructure EROT, then integrate gradient flows and selected QOTLib algorithms into an openly shareable scientific library for independent experiments and large solves. Target Hopper-or-newer NVIDIA GPUs with configurable resources, preserve a CPU path, and keep SDPLab independent.

**Architecture:** retain JAX as the common computational implementation, add explicit solver state and geometry, and permit optional native kernels or solver backends when measured performance or missing capability warrants them.

**Specification:** [architecture and numerical contract](../specs/2026-09-21-erot-flows-hpc-design.md).

**Engineering reference:** [YACHT adoption decisions](../specs/2026-09-21-yacht-conventions.md).

**Algorithm reference:** [QOTLib assessment and adoption sequence](../specs/2026-09-21-qotlib-adoption.md).

**First executable plan:** [Stage 0 cleanup and structure](2026-09-21-erot-stage-0-cleanup.md), followed by [Stage 1 numerical/hardware baselines](2026-09-21-erot-stage-1-baselines.md).

## Delivery rules

Each stage is a separately reviewable increment. Complete its entry conditions, preserve its numerical evidence, and assess its exit gate before promoting the affected backend. A later hardware investigation may start early when it resolves a blocking architectural assumption; this does not imply that dependent implementation is complete.

The first stage has a detailed task plan. Later work packages below define files, interfaces, validation, dependencies, and decision gates. Before executing each later stage, refine that package into task-sized changes against the then-current checkout and the preceding measurements. Do not write speculative CUDA integration code before choosing its operation and runtime.

Keep behavior-preserving moves, formatting, numerical corrections, performance changes, and release cleanup in separate changes. Keep `SDP-Simulation/` outside every stage. Preserve the nested gradient-flow and QOTLib checkouts and histories; establish source reuse rights before copying code for public distribution.

Calendar estimates depend on cluster access, target problem sizes, and the quantum backend decision. Stage 0 establishes the repository foundation and CPU compatibility evidence; Stage 1 produces the detailed workload inventory and hardware budget. Cleanup does not wait for cluster availability. No speedup or problem capacity below is represented as measured.

## Stage overview

| Stage | Deliverable | Depends on | Exit evidence |
|---|---|---|---|
| 0 | Repository cleanup and YACHT-inspired production foundation | Source inventory and existing CPU checks within this stage | Recoverable artifacts/history, `src/` installation, isolated wheel/sdist/CLI checks, style and test organization |
| 1 | Numerical, environment, performance, and quantum feasibility baseline | 0 | Reproducible cases, QOT/flow audits, comparison records, actual hardware facts |
| 2 | Composable dense solver core | 1 | Shared potentials/state, transformation tests, unchanged public solve results |
| 3F | Gradient flows integrated into EROT | 2 and normalization audit | Independent JKO reference checks, warm starts, explicit failure diagnostics |
| 3Q | Quantum operators and dense entropy dual solver | 2, QOT formulation audit, reuse rights before source import | Complex/unequal-dimension operator tests, independent entropy-QOT references, device state and failure diagnostics |
| 4 | Classical memory scalability | 2; flow validation uses 3F | Blocked geometry and reductions avoid dense allocation and match dense references |
| 5 | Independent cluster experiments | 3F and packaging foundation; quantum workflows use 3Q | Reproducible workers and checkpoint/resume within configurable allocations |
| 6 | Distributed individual solves | 4 and runtime work in 5 | Correct two-GPU solves, measured capacity/scaling, explicit quantum capability |
| 7 | Optional native acceleration | Relevant profiles or early capability decision | Matched-accuracy full-solve gain, memory gain, or required new capability |
| 8 | Supported integrated release | 0-6 for selected capabilities; 7 only for selected native features | Clean installation, published support matrix, examples, migration and performance reports |

Stages 3F and 3Q are separate tracks after the shared core; neither requires the other's completed algorithms. Stages 4 and 5 can progress independently once their dependencies exist. Stage 7's quantum feasibility work can begin after Stage 1 and need not wait for Stage 6. Multi-node promotion is conditional on access and interconnect evidence.

## Stage 0: repository cleanup, packaging, tests, and style

**Entry:** inventory the current checkout and establish a working CPU environment. Existing behavior is checked at the start of this stage, before moving files.

**Files:** move `erot/` to `src/erot/`; reorganize existing tests and historical artifacts; update `pyproject.toml`, CI, `.gitignore`, and README; add provenance/artifact manifests, `CONTRIBUTING.md`, `ruff.toml`, and development documentation. See the detailed cleanup plan for exact path mappings.

**Work:**

- [x] Record source revisions/status, existing CPU suite and CLI outcomes, and a small benchmark snapshot.
- [x] Inventory generated outputs, required fixtures, nested histories, and source reuse questions; archive historical artifacts with hashes rather than discarding them.
- [x] Adopt setuptools discovery under `src`, preserving public imports, CLI flags, and numerical behavior.
- [x] Isolate local reference repositories from the package and parent index; keep SDPLab untouched.
- [x] Introduce formatting/correctness checks separately from file moves and add contributor guidance.
- [x] Build wheel and sdist, rebuild from sdist, install outside the checkout, and verify API/CLI behavior without optional native/plotting dependencies.

**Exit:** recoverable source/artifact inventory, unchanged existing numerical outcomes, correct installed distributions, intentional test discovery, and a clean active layout. Neither new solver algorithms nor cluster performance evidence is required for this gate.

## Stage 1: numerical contracts and performance evidence

**Entry:** Stage 0 package/install checks pass and the original behavior is preserved.

**Files:** add numerical convention/audit documents, `docs/performance/baseline-protocol.md`, `docs/performance/quantum-feasibility.md`, reporting and legacy audit tools under `benchmarks/`, and focused unit/reference tests. Reports go under ignored `benchmark-results/stage-1/`.

**Work:**

- [x] Pin Shannon/quadratic/JKO objectives with independent small references and retain complex/unequal-dimension QOT coverage.
- [x] Reproduce or rule out the flow gradient normalization, PDHG defaults/constraints, and entropy-prox discrepancies without changing the historical checkout.
- [x] Audit QOTLib's entropy conjugates, primal recovery, complex dual differentiation, and block objective before selecting an import; record source reuse status.
- [x] Extend timing to repeated synchronized samples, scientific comparison keys, outcome checks, and isolated memory records.
- [x] Prepare configurable hardware/allocation inventory; record CPU availability and explicit GPU unavailability. Actual Hopper-or-newer memory, topology, runtime and interconnect validation awaits hardware (user-confirmed).
- [x] Assess full quantum spectral memory and native distributed eigensolver feasibility independently of classical sharding.

**Interface:** this stage characterizes installed `src/erot/` behavior without changing solver formulas. Historical audits accept explicit source paths and run in isolated environments.

**Exit:** regenerable numerical evidence, minimal reproductions of actual discrepancies, and measured or explicitly unavailable hardware records. Hardware-independent core work may proceed from local evidence; performance decisions wait for relevant measurements.

## Stage 2: pure dense core and explicit solver state

**Entry:** Stage 0 installed-package checks and Stage 1 local numerical contracts pass; outstanding findings are explicitly assigned.

**Files:** add `src/erot/solvers/state.py`, `src/erot/solvers/sinkhorn.py`, and focused solver tests; adapt `src/erot/classical.py`, `src/erot/api.py`, `src/erot/types.py`, and `src/erot/validation.py`. Add quantum resume state in `src/erot/solvers/quantum.py` only with its invariant test.

**Work:**

- [x] Define PyTree state and array-valued diagnostics, separating warm starts, exact resume state, and results.
- [x] Extract the existing multi-marginal Sinkhorn iteration and return cost-unit potentials, status, iterations, and residuals.
- [x] Add a warm-start input and documented gauge; test zero-support marginals and changed epsilon.
- [x] Separate plan evaluation/materialization from the solve result. Keep the public dense wrapper behavior.
- [x] Ensure host validation and scalar conversion do not occur inside device computation; preserve explicitly supplied array placement.
- [x] Test `jit`, `vmap`, and a short `scan` of related solves, without claiming reverse-mode differentiation through a dynamic convergence loop.
- [x] Compare dense CPU outputs with Stage 1 references; prepare optional GPU agreement/placement tests (hardware unavailable) and preserve two-marginal, rectangular, and multi-marginal support.

**Interface:** downstream flows receive potentials and diagnostics from one shared core. Numerical parameters remain dynamic where practical; shape and algorithm choices that change compilation are documented. Failure is a device status, not hidden NaNs or an untraceable Python exception.

**Exit:** standalone API results remain compatible, warm starts reach the same solution accuracy, batched/compiled composition works, and no Python synchronization remains in the core. A warm start need not reduce iterations on every arbitrary input; measure benefit on the representative adjacent-flow cases.

## Stage 3F: validated gradient flows

**Entry:** the shared Sinkhorn core works and the normalization discrepancy has a reference test.

**Files:** add `src/erot/flows/functionals.py`, `src/erot/flows/jko.py`, `src/erot/flows/pdhg.py`, `src/erot/flows/state.py`, `src/erot/optim/prox.py`, and flow/reference tests. Add `examples/heat_flow.py`, `examples/quadratic_flow.py`, and flow documentation. Preserve legacy source provenance.

**Work:**

- [ ] Port the Sinkhorn JKO outer iteration onto the shared core and apply the independently verified cost/potential normalization.
- [ ] Provide entropy, potential, and quadratic functionals with explicit value/gradient/prox capabilities and cell-volume conventions.
- [ ] Consolidate PDHG into one device implementation. Verify the norm of the actual coupled operator, step-size bound, extrapolation semantics, and positive mass handling.
- [ ] Validate entropy prox using its scalar optimality equation over extreme inputs; use a stable method instead of overflowing `exp(z/alpha)`.
- [ ] Retain the initial SGD option using JAX operations; keep alternate optimizers optional if added. Do not force optimizer objects or Flax into the base API.
- [ ] Return flow state, accepted-step count, physical time, inner work, mass/constraint errors, and failure reason. Expose snapshots independently from full solver checkpoints.
- [ ] Test unregularized and entropic JKO against their own reference objectives; verify failed inner steps do not silently advance time.

**Interface:** `jko_step` evolves one state; a chunk runner executes a fixed number of steps with explicit output selection. Inputs include the energy, geometry, time step, backend, numerical tolerances, and optional compatible state. Native acceleration is not needed to ship this interface.

**Exit:** at least one entropy-flow and one quadratic-energy case match independent discrete references; cost convention, mass conservation, feasibility, and failure behavior are tested. A heat-equation study records boundary conditions and refinement parameters rather than comparing only to an equilibrium distribution.

## Stage 3Q: selected quantum algorithms from QOTLib

**Entry:** shared core/state contracts, an explicit QOT formulation, independent small references, and source reuse rights for any copied implementation.

**Files:** add `src/erot/operators/quantum.py`, `src/erot/solvers/quantum_entropy.py`, quantum unit/reference tests, and a small entropy-QOT example. Preserve `src/erot/quantum.py` compatibility. The [QOTLib assessment](../specs/2026-09-21-qotlib-adoption.md) details Q1/Q2 and later experimental candidates.

**Work:**

- [ ] Implement partial-trace and adjoint contracts with complex inputs and unequal subsystem dimensions.
- [ ] Define the spectral entropy primal/dual pair and trace/mass convention; distinguish trace-exponential and trace-constrained log-partition formulations.
- [ ] Adapt the dual solve into pure JAX state and diagnostics, optional optimizer dependencies, bounded history, and explicit convergence/failure rules.
- [ ] Validate primal recovery, marginal feasibility, stationarity, objective agreement, and complex derivatives independently.
- [ ] Keep block PDHG/chordal and Lanczos extensions experimental until their own formulation, equivalence/approximation, and performance gates pass.

**Exit:** a tested dense entropy-QOT capability with documented limits and attribution; no claim of quantum gradient flow, matrix-free scaling, or distributed spectral support without separate evidence.

## Stage 4: memory-aware classical transport

**Entry:** dense results and state semantics are stable. Integration checks use Stage 3F when a flow is benchmarked.

**Files:** add `src/erot/geometry/dense.py`, `src/erot/geometry/pointcloud.py`, `src/erot/geometry/reductions.py`, and `src/erot/geometry/plan.py`; extend the Sinkhorn core, benchmark cases, and geometry/reference tests. Grid geometry is a separate increment under `src/erot/geometry/grid.py` if justified.

**Work:**

- [ ] Implement block cost evaluation and stable streamed log-sum-exp accumulation for both marginal directions.
- [ ] Handle non-divisible tiles, zero supports, all-masked tiles, and very small epsilon without converting padding to transport mass.
- [ ] Compute residuals and objectives without materializing a full plan; offer explicit plan blocks and transport application.
- [ ] Inspect compiled computation and measure peak live memory to prove that no hidden dense cost/coupling remains.
- [ ] Compare dense and blocked problems at matched objective/residual tolerance and measure block choices on each available GPU profile.
- [ ] Add separable/convolution geometry only with independent dense-equivalence tests for its metric and boundary conditions.

**Exit:** memory grows according to stored geometry, potentials, and chosen tiles instead of a full `n*m` coupling for the supported point-cloud path. Generic arithmetic remains quadratic and is documented. Existing arbitrary dense and multi-marginal paths remain available with their original capacity limits.

## Stage 5: reproducible cluster experiments

**Entry:** installed flow workflows pass; core runtime configuration is explicit.

**Files:** add `experiments/run.py`, `experiments/configs/`, `src/erot/runtime/checkpoint.py`, `src/erot/runtime/environment.py`, `hpc/run_array.sbatch`, `hpc/profiles/`, and restart/integration tests. Extend CLI only where needed to invoke an installed workflow.

**Work:**

- [ ] Define schema-versioned run configurations and unique experiment identity; record seeds, input identity, source/runtime versions, dtype, and output policy.
- [ ] Implement chunked execution, atomic checkpoints, interrupted-run recovery, and validation against incompatible restart input.
- [ ] Start one worker per allocated GPU by default; derive CPU/thread budgets from scheduler affinity and explicit resource configuration. Reject oversubscribed settings.
- [ ] Provide configurable Hopper-or-newer profiles without hard-coded accounts, GPU counts, CPU limits, personal paths, or assumptions about peer links. The old two-GPU/16-core allocation is only an example.
- [ ] Store logs and results per run; aggregate scientific results separately from numerical execution.
- [ ] Measure job throughput including startup/compilation and checkpoint overhead. Only batch same-shape trials when beneficial.

**Exit:** one- and two-worker smoke cases are independent and reproducible, interruption/resume matches uninterrupted output within tolerance, and no output collisions or CPU oversubscription occur. Worker configuration supports larger allocations; claim measured throughput only for tested counts. Experiment throughput is evaluated separately from one-problem scaling.

## Stage 6: a single distributed solve

**Entry:** Stage 4 avoids dense materialization; Stage 5 supplies process configuration and checkpoint ownership. Hardware topology and supported runtime are known.

**Files:** add `src/erot/runtime/distributed.py`, `src/erot/solvers/distributed_sinkhorn.py`, `hpc/run_distributed.sbatch`, and `tests/distributed/`; extend geometry and reporting with global/local shape and communication information.

**Work:**

- [ ] Verify two local GPUs under one process, followed by two processes owning one GPU each.
- [ ] Implement the row-partitioned Sinkhorn communication contract from the specification, including stable global reductions and globally consistent convergence decisions.
- [ ] Test uneven partitions, zero support on one shard, and rank-consistent iteration/termination behavior.
- [ ] Generate/load local input blocks, preserve global array semantics, and checkpoint all required shards without gathering the full plan.
- [ ] Measure speedup on problems fitting one GPU and capacity on larger problems. Report per-GPU peak memory and communication share.
- [ ] Evaluate distributed dense PDHG separately; do not infer its memory behavior from the implicit Sinkhorn implementation.
- [ ] Promote to two nodes only when interconnect and launch tests pass. Keep unsupported or unmeasured platforms explicit.

**Quantum branch:** use Stage 1 feasibility evidence to select a supported single-GPU limit, a distributed native eigensolver integration, or a separately scoped structured formulation. Release notes distinguish these outcomes; they must not imply that sharding `eigh` is solved by classical transport sharding or a single Lanczos eigenpair.

**Exit:** numerical equivalence and collective correctness pass on real target hardware. A distributed mode is justified by measured speed or capacity. An illustrative investigation target is 1.3x speedup on two GPUs for sufficiently large compute-dominated cases; failure to meet it triggers analysis, not an invented performance claim.

## Stage 7: optional native operations or solver backends

**Entry:** a profile identifies a candidate or a required capability cannot be supplied adequately. Quantum capability investigation may bring this stage forward.

**Files:** introduce only the selected component under `native/` and a corresponding `src/erot/backends/` adapter, native correctness tests, build configuration, and compatibility documentation. Do not scaffold every possible backend.

**Work:**

- [ ] Calculate a plausible full-solve benefit using the measured runtime fraction and Amdahl's law.
- [ ] Compare algorithm/representation improvements and existing optimized libraries before writing a new kernel.
- [ ] Compare regular JAX, Pallas Mosaic, and CUDA/C++ for the actual operation. Validate selected Hopper-or-newer architectures, required dtypes, runtime versions, and fallback; never assume a successor GPU supports every prior specialization.
- [ ] For an existing Fortran library, measure the complete interface and transfer/communication cost; use it where the algorithm or proven implementation supplies a benefit.
- [ ] Test stream ownership, aliasing, layouts, nonfinite/error paths, batching, sharding, and any advertised derivatives. Reject unsupported transformations explicitly.
- [ ] Validate optional installation, platform metadata, runtime version compatibility, and fallback behavior.
- [ ] Retain only implementations meeting the specification's full-solve speed/memory gate or an explicit capability requirement.

**Exit:** at least 1.5x matched-accuracy full-solve speedup, at least 25% measured peak live-memory reduction, or a documented required capability gain; numerical checks and the support matrix pass. No native implementation is also a valid outcome if the evidence favors JAX.

## Stage 8: release and migration

**Entry:** selected capabilities have numerical and hardware evidence. Optional unimplemented research work is not a release dependency.

**Files:** update `README.md`, `docs/migration-0.2.md`, `docs/development.md`, API/numerical/HPC guides, changelog, examples, package metadata, and release workflows.

**Work:**

- [ ] Document supported objectives, derivative support, precision modes, capacity limits, and actual GPU/distributed validation. Separate validated configurations from experimental and untested hardware.
- [ ] Verify clean wheel/sdist installation and execute documented CPU examples and CLI workflows. Extend automated content inspection to both direct and sdist-derived wheels (Stage 0 review follow-up).
- [ ] Publish reproducible benchmark configurations and clearly separate measurements from extrapolations.
- [ ] Provide migration mappings from `jko_lab` into `erot.flows` and adopted QOTLib components into EROT, with scientific behavior changes, source revisions, reuse rights, and attribution recorded.
- [ ] Remove redundant imported code only after every retained capability and historical source is accounted for; treat repository relocation and remote changes as separate explicit operations.
- [ ] Keep SDPLab independent; do not introduce a shared runtime dependency merely because both projects run on the same cluster.

**Exit:** an independently installable EROT release with reproducible examples, clear support limits, and a recoverable migration history. No publication or remote repository mutation is part of the current planning request.

## Later research work packages

| Direction | Prerequisite | Required evidence |
|---|---|---|
| Direct KL-proximal/generalized Sinkhorn JKO | Stage 3F energy contracts | Applicable KL prox, independent discrete objective, comparison with nested optimization |
| Dynamic density/flux PDHG | Explicit spatial/time discretization | Boundary conditions, reference PDE convergence, memory and communication scaling |
| QOTLib block/chordal or low-rank quantum algorithms | Stage 3Q operators and stated structure | Dense equivalence where claimed; approximation error/optimization guarantees otherwise; measured end-to-end memory and convergence |
| End-to-end differentiation | Stable solve and flow contracts | Finite-difference checks, gauge treatment, convergence assumptions, memory/cost analysis |

These packages require their own mathematical designs. They do not enter the first integration release through a silent change of backend.
