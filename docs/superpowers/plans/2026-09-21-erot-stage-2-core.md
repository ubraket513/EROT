# Stage 2: composable dense solver core

Execute inline using `superpowers:executing-plans`. The user already authorized
implementation, local commits and subsequent stages. Hardware runs remain
unavailable; prepare optional GPU checks and keep CPU evidence explicit.

**Status:** implemented and reviewed on CPU. See [verification](../../development/stage-2-validation.md). Actual GPU checks remain unexecuted.

## Contract and implementation decisions

Preserve `erot.solve`, `SolverConfig`, dense `SolveResult`, CLI and
`erot.classical.shannon_sinkhorn` return signatures. Add device APIs under
`erot.solvers`. Use NamedTuple PyTrees for portability to the declared JAX floor;
keep scalar parameters as arrays and structural shapes static. No optional
optimizer dependency enters the base installation.

`SinkhornState(potentials, iterations)` contains cost-unit potentials and the
cumulative completed sweep count. Exact resume is valid only for identical
cost, marginals, epsilon and dtype. `SinkhornWarmStart(potentials)` is a distinct
object for related problems; it resets the count and permits changed epsilon.
Normalize the first k-1 potentials to their positive-support marginal weighted
mean zero and compensate in the final potential. Zero support uses -infinity;
newly positive support in a warm start resets nonfinite entries to zero.
Do not accept NaNs on positive support as an exact resume.

`SolverDiagnostics(error, iterations, status)` is entirely array-valued.
Status codes distinguish converged, iteration limit, invalid input and numerical
failure. The core accepts nonnegative additional sweep budgets, including zero,
and recomputes diagnostics at the returned state. Dynamic validation must not
introduce a host conversion. Static shape misuse may raise a Python ValueError.
The validated host wrapper may synchronize deliberately.

`solve_sinkhorn(cost, marginals, epsilon, tolerance, max_iterations,
state=None, warm_start=None)` returns state and diagnostics without a coupling.
`materialize_plan(cost, potentials, epsilon)` computes the dense tensor only
when requested. This initial dense iteration still forms dense temporary plans;
Stage 4 removes those temporaries for point-cloud geometry. Do not claim memory
scalability before that change. No reverse-mode claim through `lax.while_loop`.

## Task 1: state, pure Sinkhorn and numerical regression

- [x] Add `src/erot/solvers/state.py`, `sinkhorn.py`, and `__init__.py`.
- [x] Write tests first for independent rectangular objective/coupling agreement,
  multi-marginal zero support/non-unit mass, cost-unit potential derivative,
  gauge, changed-epsilon warm starts, zero/one budgets, nonfinite input/status,
  and exact resumed sweeps versus uninterrupted sweeps.
- [x] Implement cyclic log-domain updates using other-axis potentials directly;
  avoid subtracting -infinity for inactive support. Normalize gauge at each
  sweep so resume and uninterrupted operations have identical arithmetic.
- [x] Separate plan materialization; do not retain a dense coupling in state.
- [x] Run focused tests and the existing independent numerical references.

## Task 2: transformations and compatibility integration

- [x] Write `jit`, mixed-convergence `vmap`, and short related-problem `scan`
  tests. Include adjacent-marginal warm-start work measurement without requiring
  every warm start to save iterations.
- [x] Replace only the existing Shannon wrapper implementation with a call to
  the core and explicit materialization. Preserve its three-array result.
- [x] Keep validated host conversion separate. For `device='auto'`, preserve a
  single explicitly placed input device when all supplied JAX arrays agree;
  reject ambiguous multi-device arrays at the host API with guidance to the
  pure API. An explicitly requested device remains authoritative.
- [x] Reject nonfinite config scalars at the host boundary; maintain existing
  positive-iteration host contract while the device core supports zero work.
- [x] Document dynamic parameters, status codes, gauge and resume identity.
- [x] Run compatibility/CLI tests and CPU references; add opt-in GPU placement
  coverage, recording its skip until hardware is available.

## Task 3: quantum quadratic resume

- [x] Add `src/erot/solvers/quantum.py` with the complete Dykstra coupling and
  three corrections plus cumulative count. Do not warm-start corrections across
  a changed problem. Preserve dense quantum wrapper output.
- [x] Test exact resumed versus uninterrupted sweeps on complex unequal local
  dimensions, then run existing PSD/marginal/CVXPY references.
- [x] Explicitly distinguish quantum resume from a snapshot of the coupling.

## Task 4: review and handoff

- [x] Run the full suite, Ruff checks, installed smoke appropriate to new modules,
  and inspect the diff for accidental SDPLab or historical-source changes.
- [x] Commit important milestones locally. Request the execution skill's fresh
  final review, fix material findings, record deferred minor observations.
- [x] Refine Stage 3F against these actual APIs and proceed without asking again.

Acceptance: existing dense results remain numerically compatible; state resumes
correctly; warm starts converge under the same mathematical contract; composed
JAX operations use no Python scalar conversion or host copy in the core.
