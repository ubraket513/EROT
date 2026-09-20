# EROT Stage 1: numerical and hardware baselines implementation plan

License update (2026-09-21): the user confirmed Apache-2.0 for both QOTLib and numerical-gradient-flows. Earlier missing-license observations below describe the inspected trees, not an outstanding permission question. Preserve attribution and applicable notices when adapting source.

> **For agentic workers:** Use `superpowers:executing-plans` for inline implementation, or `superpowers:subagent-driven-development` if that execution method is selected. Implement one reviewable task at a time. Checkboxes below describe future work, not work completed while preparing this plan.

**Goal:** after Stage 0 cleanup, establish reproducible numerical contracts, flow/QOTLib characterization, and performance evidence on available Hopper-or-newer GPUs before numerical integration or optimization.

**Architecture:** keep the existing solver algorithms intact during baseline collection. Add independent reference tests and repository-only measurement tools. Report legacy discrepancies explicitly and use measured capability gaps to decide whether quantum native integration must begin early.

**Tech stack:** Python >=3.11, the existing JAX/NumPy package, pytest, optional CVXPY/SciPy reference dependencies, optional historical Flax/Optax dependencies in an audit environment, and target-cluster GPU tools.

**Spec:** [architecture and numerical contract](../specs/2026-09-21-erot-flows-hpc-design.md); read alongside [YACHT conventions](../specs/2026-09-21-yacht-conventions.md), [QOTLib assessment](../specs/2026-09-21-qotlib-adoption.md), and the [stage dependencies](2026-09-21-erot-flows-hpc-roadmap.md). The [Stage 0 cleanup plan](2026-09-21-erot-stage-0-cleanup.md) runs first.

## Global constraints

- Merge reusable numerical-gradient-flows functionality into the EROT distribution; keep SDPLab as the independent second repository.
- Do not modify or relocate `SDP-Simulation/` as part of this plan.
- Target NVIDIA Hopper or newer for GPU optimization; retain a CPU installation and numerical reference path. Publish tested architecture/runtime combinations rather than promising support for every future GPU.
- Make GPU count, CPUs per task, memory, node count, device links, and scheduler settings configurable. The earlier one-/two-GPU, 16-core allocation is an example profile, not a library limit or default entitlement.
- Preserve the existing `erot.solve`, `SolverConfig`, CLI, and dense result behavior during the additive migration.
- Preserve the `src/` package layout and installed-artifact guarantees established in Stage 0.
- Retain Python >=3.11. Initially retain the declared JAX >=0.4.30 and NumPy >=1.26 floors; do not imply that these floors support future distributed or native extensions.
- Before adding version-sensitive sharding or FFI code, establish a tested JAX/jaxlib/CUDA combination and update the supported floor and environment lock together.
- Keep JAX and NumPy as the mandatory numerical dependencies. Plotting, experiment optimization, independent reference solvers, and native extensions remain optional.
- Support float64/complex128 reference computations and separately validated float32/complex64 execution; do not silently reduce precision.
- Use the same mathematical objective and accuracy requirements for every performance comparison.
- Keep original source provenance and attribution. Preserve the nested gradient-flow and QOTLib histories and local changes; establish reuse rights before importing their source into a public distribution.

Stage 1 uses the installed `src/erot/` layout and reorganized tests from Stage 0. It adds tests, reporting tools, and documentation; scientific corrections discovered here are implemented in their assigned subsequent solver/flow tasks with the original failure preserved. Historical source locations are explicit audit inputs, not runtime dependencies of EROT.

## Review focus

1. Zero support and non-unit equal mass: classical OT preserves the input mass and does not invent transport on zero support. Covered in Task 2.
2. A plausible but incorrectly scaled JKO gradient: compare a potential derivative to an independently defined optimized objective. Covered in Tasks 2 and 3.
3. False performance wins from failed convergence or changed precision/objective/hardware: invalid or incomparable records are rejected. Covered in Task 4.
4. Quantum tests accidentally covering only real/full-rank inputs or mismatched regularizers: retain complex/rank-deficient cases and audit QOTLib's objectives. Covered in Tasks 1, 2, 3Q, and 5.
5. Misleading memory or scaling results: distinguish fresh-process peak measurements, capacity estimates, device links, and genuinely available hardware. Covered in Tasks 4 and 5.

## Task 1: preserve provenance and establish the baseline environment

**Files:**

- Extend `docs/architecture/source-provenance.md` from Stage 0.
- Create `docs/performance/baseline-protocol.md`.
- Write machine reports under ignored `benchmark-results/stage-1/`.
- Inspect, but do not modify, the original nested repositories.

**Interfaces:** consumes the source checkout and cluster allocation; produces recorded revisions, dependency versions, original test output, and reproducible commands.

- [ ] **Record source state before changing files.** Run these individually from the EROT root; record stdout and exit status in the provenance document.

```bash
git status --short
git rev-parse HEAD
git -C numerical-gradient-flows status --short
git -C numerical-gradient-flows rev-parse HEAD
git -C QOTLib/QOTLib status --short
git -C QOTLib/QOTLib rev-parse HEAD
```

Record the YACHT reference revision `48c86a09dee58f693f1223eb2b1a77764a62d9bd`, QOTLib revision `bd534c61aeae082892b9b2421db153beb8e5c804`, the origin/reuse status of candidate code, and whether each source tree has local modifications. Substitute explicit external source paths if these clones are elsewhere. Do not use `git add .` or fold nested `.git` directories into the parent repository.

- [ ] **Select an isolated CPU development environment and install existing declared dependencies.** These commands are to be executed after implementation is authorized; none were run while preparing the plan.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[test,plot]'
.venv/bin/python -m pip freeze
.venv/bin/python -m pytest
.venv/bin/erot --help
```

If `.venv` already belongs to the project, inspect and reuse it or choose a fresh task-specific path; do not overwrite an unrelated environment. Use the selected interpreter consistently in every later command. Keep its exact dependency report with the baseline, without claiming that an untested minimum dependency combination passes.

- [ ] **Run the existing small benchmark unchanged.**

```bash
.venv/bin/python benchmarks/benchmark_solvers.py --device cpu --classical-sizes 16 --quantum-sizes 2 --output benchmark-results/stage-1/original-cpu.json
```

Expected: a JSON record containing the current compile-and-run and warm fields. Record nonconvergence or failures rather than modifying the solver to obtain a baseline. The full original suite includes the complex quantum and optional independent solver tests; record skips and their reasons.

- [ ] **Record the baseline review outcome.** Separate reproduced failures from unsupported environments. Create a focused local documentation commit only if committing is part of the chosen execution workflow; otherwise leave this task's files reviewable as a bounded diff.

## Task 2: pin objective conventions with independent tests

**Files:**

- Create `docs/numerics/conventions.md`.
- Create `tests/reference/test_numerical_contract.py`.
- Extend `tests/unit/test_classical.py` and `tests/unit/test_quantum.py` only for uncovered invariants.

**Interfaces:** consumes the existing `erot.solve` and `SolverConfig`; produces objective definitions and reference assertions that the later core/flow migration must preserve.

- [ ] **Document the actual objectives from the specification.** Include coupling entropy with `-1`, cost-unit potentials, the squared-distance `1/(2*time_step)` JKO factor, quadratic regularization, cell-mass semantics, and quantum partial traces. State that epsilon changes the mathematical approximation.

- [ ] **Add a small entropic reference test.** Use positive masses and a rectangular cost so the reference is neither a trivial symmetric solution nor a check of marginals alone. A concrete test body is:

```python
import numpy as np
import pytest

import erot


def shannon_value(cost, plan, epsilon):
    positive = plan > 0
    entropy = np.sum(plan[positive] * (np.log(plan[positive]) - 1.0))
    return float(np.sum(cost * plan) + epsilon * entropy)


def reference_shannon(cost, a, b, epsilon):
    cp = pytest.importorskip("cvxpy")
    plan = cp.Variable(cost.shape, nonneg=True)
    row_constraint = cp.sum(plan, axis=1) == a
    col_constraint = cp.sum(plan, axis=0) == b
    objective = cp.Minimize(
        cp.sum(cp.multiply(cost, plan))
        - epsilon * cp.sum(cp.entr(plan))
        - epsilon * cp.sum(plan)
    )
    problem = cp.Problem(objective, [row_constraint, col_constraint])
    problem.solve(solver="CLARABEL")
    assert problem.status == "optimal"
    return float(problem.value), np.asarray(plan.value), -row_constraint.dual_value


@pytest.mark.reference
def test_shannon_objective_matches_independent_reference():
    cost = np.array([[0.0, 0.7], [0.3, 0.1], [0.9, 0.2]])
    a = np.array([0.2, 0.5, 0.3])
    b = np.array([0.4, 0.6])
    epsilon = 0.7
    result = erot.solve(
        cost,
        [a, b],
        problem="classical",
        regularizer="shannon",
        method="sinkhorn",
        config=erot.SolverConfig(
            epsilon=epsilon, tolerance=1e-10, max_iterations=10000,
            dtype="float64", device="cpu",
        ),
    )
    expected_value, expected_plan, _ = reference_shannon(cost, a, b, epsilon)
    assert result.converged
    np.testing.assert_allclose(result.coupling, expected_plan, atol=2e-5)
    assert shannon_value(cost, np.asarray(result.coupling), epsilon) == pytest.approx(
        expected_value, abs=2e-6
    )
```

- [ ] **Run the test and inspect its failure or success.**

```bash
.venv/bin/python -m pytest tests/reference/test_numerical_contract.py -v
```

This is characterization of existing behavior, so a new test can pass immediately. Do not manufacture an expected failure. If it fails, verify the independent oracle's feasibility/accuracy and retain the actual discrepancy for the appropriate correction stage.

- [ ] **Add a mass-preserving potential derivative test.** In the same test module, reuse the exact `cost`, `a`, `b`, and `epsilon` above. Recover cost-unit potentials from the positive converged coupling using the stationarity identity, then compare against a finite difference of the independent reference objective:

```python
@pytest.mark.reference
def test_sinkhorn_potential_derivative_matches_reference():
    cost = np.array([[0.0, 0.7], [0.3, 0.1], [0.9, 0.2]])
    a = np.array([0.2, 0.5, 0.3])
    b = np.array([0.4, 0.6])
    epsilon = 0.7
    result = erot.solve(
        cost, [a, b], problem="classical", regularizer="shannon",
        method="sinkhorn",
        config=erot.SolverConfig(
            epsilon=epsilon, tolerance=1e-10, max_iterations=10000,
            dtype="float64", device="cpu",
        ),
    )
    assert result.converged
    stationarity = cost + epsilon * np.log(np.asarray(result.coupling))
    g = stationarity[0, :]
    f = stationarity[:, 0] - g[0]
    np.testing.assert_allclose(f[:, None] + g[None, :], stationarity, atol=2e-7)
    direction = np.array([1.0, -1.0, 0.0])
    h = 1e-3
    plus, _, _ = reference_shannon(cost, a + h * direction, b, epsilon)
    minus, _, _ = reference_shannon(cost, a - h * direction, b, epsilon)
    derivative = (plus - minus) / (2 * h)
    assert np.dot(f, direction) == pytest.approx(derivative, rel=2e-3, abs=1e-4)
```

Check `h=5e-4` as a diagnostic if the comparison is ambiguous; do not loosen tolerances merely to hide a factor of two. The test operates on strictly positive support and intentionally cancels potential gauge constants.

- [ ] **Add the equal-mass boundary case and retain quantum cases.** For cost `[[0, 1], [2, 0]]`, `a=[2, 0]`, and `b=[0.5, 1.5]`, the only feasible plan is `[[0.5, 1.5], [0, 0]]`; compare the result against it with `atol=1e-8` in float64. Existing multi-marginal zero-support, unequal quantum subsystem sizes, complex pure marginals, and CVXPY quadratic reference tests must remain in the baseline.

- [ ] **Review the scientific contract independently of package structure.** Run focused classical/quantum/reference tests and preserve results. Stage 0 already moved files to `src/`; do not combine another layout change or altered iteration formula with this audit.

## Task 3: reproduce legacy flow discrepancies without concealing them

**Files:**

- Create `benchmarks/audit_legacy_flow.py`.
- Create `docs/numerics/legacy-flow-audit.md`.
- Put audit results under `benchmark-results/stage-1/legacy/`.

**Interfaces:** consumes the original `jko_lab` code from an explicit `--source-path` in a separate audit environment; produces JSON records with `check`, `status`, input/configuration, measured values, and any exception type/message. Status is `passed`, `mismatch`, `error`, or `unavailable`. Unavailable dependencies are never counted as a pass. The ordinary EROT test suite must not depend on this checkout.

- [ ] **Create a separate historical audit environment.** Install the historical project and its actual imported dependencies without adding them to EROT's mandatory requirements:

```bash
python -m venv .venv-legacy-audit
.venv-legacy-audit/bin/python -m pip install -e '.[test]' -e ./numerical-gradient-flows flax optax
.venv-legacy-audit/bin/python -m pip freeze
```

Record the resolved versions. If historical compatibility requires a different environment, report the incompatibility and pin a working audit combination before interpreting numerical results.

- [ ] **Audit the potential factor using the same positive case as Task 2.** The script enables float64 before creating arrays, builds `SinkhornJKO(C, rho0=b, eta=0.1, epsilon=0.7, sinkhorn_iters=10000, tol=1e-10)`, and calls `compute_W2_gradient(a, b)`. Evaluate the independent optimized transport value at `a +/- h*direction` using the CVXPY expression in Task 2. Compare the returned gradient dotted with `direction` to this derivative, and separately report `f` dotted with that direction. Record all three numbers and their residuals; this identifies a factor error without assuming it.

The audit uses this explicit comparison logic:

```python
def derivative_audit(predicted, reference, *, rtol=2e-3, atol=1e-4):
    import numpy as np

    valid = bool(np.isfinite(predicted) and np.isfinite(reference))
    matches = valid and bool(np.isclose(predicted, reference, rtol=rtol, atol=atol))
    return {
        "status": "passed" if matches else "mismatch",
        "predicted": float(predicted) if np.isfinite(predicted) else None,
        "reference": float(reference) if np.isfinite(reference) else None,
        "rtol": rtol,
        "atol": atol,
    }
```

- [ ] **Audit PDHG defaults and constraints.** Use `x=[0, 0.5, 1]`, `C_ij=(x_i-x_j)^2`, initial mass `[0.2, 0.5, 0.3]`, `eta=0.1`, and `partial(proxF_quadratic, b=target, lam=1.0)` with `target=[0.3, 0.3, 0.4]`. Exercise `PrimalDualJKO.take_step` once with default step sizes and once with explicit `tau=sigma=0.01`, using 2000 inner iterations. Also exercise the functional `pdhg_jko` entry point, which exposes unnormalized `state.rho`; the class returns a normalized density and must not be described as exposing the raw iterate. Record actual exceptions, `norm(row_sums(state.pi)-state.rho)`, `norm(col_sums(state.pi)-rho_k)`, total mass, and the objective before normalization. Compare to a CVXPY problem with `pi >= 0`, `rho = row_sums(pi)`, `col_sums(pi)=rho_k`, and objective `0.1 * 0.5 * sum_squares(rho-target) + 0.5 * sum(C*pi)`. A normalized density alone is not evidence of a correct step.

- [ ] **Audit entropy-prox stability.** Evaluate existing `proxF_entropy` with `z=[-10, -1, 0, 1, 10]` and `alpha` in `[0, 1e-4, 0.1, 1]`. For positive alpha, compare to a scalar root solve of `x-z+alpha*log(x)=0` in log coordinates, and record nonfinite outputs. At alpha zero, specify the domain convention explicitly: for the nonnegative-domain entropy functional the limiting proximal map is projection onto the nonnegative half-line. Do not accept negative outputs as the correct domain-limited prox by accident.

- [ ] **Run the audit and assign discrepancies.**

```bash
.venv-legacy-audit/bin/python benchmarks/audit_legacy_flow.py --source-path numerical-gradient-flows --output benchmark-results/stage-1/legacy/audit.json
```

The proposed CLI exits nonzero when a check is mismatched, errored, or unavailable, while retaining the complete report. Document expected historical failures separately from the normal CI suite. Stage 3F owns verified corrections during migration; the original checkout remains available for reproduction.

## Task 3Q: characterize quantum candidates before adoption

**Files:** create `docs/numerics/qotlib-audit.md`; add optional `benchmarks/audit_qotlib.py` only for executable checks. Keep reports under `benchmark-results/stage-1/qotlib/`. Do not copy QOTLib implementation into `src/erot/` during this task.

**Interfaces:** an explicit source path and revision identify the audited checkout. Use the same outcome vocabulary as the flow audit. Install only the dependencies needed for the selected checks in a separate environment; QOTLib's broad requirements file is not an EROT runtime specification.

- [ ] **Record provenance and rights.** Recheck the license/permission status before any source adaptation. Preserve the supplied checkout. An unresolved copying permission is reported distinctly from an unavailable dependency or failed numerical test.
- [ ] **Check the operator contract independently.** For two and three small subsystems, compare partial traces to a NumPy tensor-index calculation and verify `Re <A(X), U> = Re <X, A*(U)>`. Include complex Hermitian inputs. Record QOTLib's equal-local-dimension limitation and retain EROT's unequal-dimension references.
- [ ] **Specify the entropy formulations.** Derive the sum-of-exponentials and trace-constrained log-partition conjugates with their constants. Use small diagonal density matrices and a separately solved classical entropy problem to check primal/dual values. Compare trace and marginal residuals before and after QOTLib's optional primal normalization. Do not treat trace one as full QOT feasibility.
- [ ] **Audit dual optimization.** Check real and imaginary directional derivatives of the chosen real-valued dual objective against finite differences. Exercise zero/one iteration caps, nonfinite values, and deliberate nonconvergence; record termination status and residuals at the returned iterate. Synchronize results for timing and record which state is sufficient for restart.
- [ ] **Audit block PDHG as its actual problem.** Expose the hard-coded regularization and omitted logged penalty in the report. On the smallest supported blocks, compare constraints/objective against an independently defined dense reference under the same regularization. Record fixed-iteration outcomes without asserting convergence from the loop count.
- [ ] **Classify spectral scalability.** Record dense allocations in slack construction and primal recovery. For Lanczos, compute `norm(Ax-lambda*x)` independently and compare small real/complex cases to dense eigenvalues. A successful eigenpair test is not evidence for full PSD projection equivalence.
- [ ] **Assign adoption decisions.** Map operator/entropy work to Stage 3Q (Q1/Q2), block/chordal work to experimental Q3, and matrix-free spectral work to Q4. Only relevant Q1/Q2 findings block the dense entropy solver; experimental research does not block cleanup, classical core work, or flows.

Expected output is a reproducible assessment with measured outcomes where execution is possible, remaining unknowns, and a specific mathematical contract for each accepted algorithm. Historical solver scripts are not automatically part of supported EROT tests.

## Task 4: make benchmark comparisons scientifically valid

**Files:**

- Create `benchmarks/__init__.py` and `benchmarks/reporting.py`.
- Modify `benchmarks/benchmark_solvers.py` and `benchmarks/README.md`.
- Create `tests/unit/test_benchmark_reporting.py`.

**Interfaces:**

- `summarize_timings(samples_seconds: Sequence[float]) -> dict[str, object]` validates and summarizes synchronized samples.
- `validate_record(record: Mapping[str, object]) -> None` rejects a nonconverged record or nonfinite required metrics for performance comparison.
- `assert_comparable(candidate: Mapping[str, object], baseline: Mapping[str, object]) -> None` rejects mismatched scientific/environment contracts.
- Report schema version 2 preserves original first-call/warm fields for readers where practical and adds sample lists, provenance, outcome, and comparison identity. Do not silently compare a legacy record lacking that identity.

- [ ] **Write and run the reporting tests first.** Include these meaningful cases:

```python
import pytest

from benchmarks.reporting import summarize_timings


def test_timing_summary_uses_median_and_keeps_samples():
    samples = [1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0]
    result = summarize_timings(samples)
    assert result["median_seconds"] == 4.0
    assert result["min_seconds"] == 1.0
    assert result["max_seconds"] == 9.0
    assert result["samples_seconds"] == samples


@pytest.mark.parametrize("samples", [[], [float("nan")], [float("inf")], [-1.0], [0.0]])
def test_invalid_timing_samples_are_rejected(samples):
    with pytest.raises(ValueError):
        summarize_timings(samples)
```

Run `.venv/bin/python -m pytest tests/unit/test_benchmark_reporting.py -v`. Expected before implementation: missing reporting module/function. Confirm that the failure is the intended missing behavior, not an unrelated environment problem.

- [ ] **Implement the summary helper and validate outcomes.** A complete summary implementation is:

```python
import math
from statistics import median


def summarize_timings(samples_seconds):
    samples = [float(value) for value in samples_seconds]
    if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("timing samples must be nonempty, finite, and positive")
    return {
        "median_seconds": median(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "samples_seconds": samples,
    }
```

`validate_record` requires true convergence, finite objective and residual, and finite positive timings. A numerical-reference validation field records which case family has been checked; it does not manufacture an optimality certificate for a large solve. Test false convergence and NaN objective/residual explicitly. Legitimate negative-infinite potentials on zero support are not themselves an invalid objective or residual.

- [ ] **Define and test comparison identity.** Require equality of schema, problem/case digest, objective convention, dtype, geometry/shape, epsilon, tolerance, iteration limit, output policy, hardware profile, and dependency/runtime profile. Exclude the source revision from required equality so two implementations can be compared, but retain both revisions in the report. For flow records also compare energy, time step, physical horizon, inner tolerance, and checkpoint/output policy. Parameterize tests to change each identity field individually and require an explicit incompatibility error.

- [ ] **Extend timing in the benchmark driver.** Keep the first invocation as `first_call_seconds` (trace/compile plus execution, not pure compiler time). Execute one more warm-up, then seven measured calls, synchronizing every call before stopping its timer. Set the compatibility field `warm_seconds` to the median and retain all raw samples. Require every measured solve to converge. Use a fresh process for a new per-case peak-memory experiment; keep allocator reservation and live-memory fields separately named and nullable. Query the actual participating device(s), not unconditionally `jax.devices()[0]`, which may differ from the requested benchmark device.

- [ ] **Preserve both repository invocation forms.** Support `python -m benchmarks.benchmark_solvers` and the existing direct script command. If importing the new helper, use explicit package/direct-script branches based on `__package__`; do not catch arbitrary import failures and silently fall back.

- [ ] **Run reporting tests and a smoke measurement.**

```bash
.venv/bin/python -m pytest tests/unit/test_benchmark_reporting.py -v
.venv/bin/python -m benchmarks.benchmark_solvers --device cpu --classical-sizes 16 --quantum-sizes 2 --output benchmark-results/stage-1/cpu-v2.json
```

Expected: versioned records with seven warm samples, consistent objectives/outcomes, and explicit environment identity. Existing first-call/warm readers continue to work where documented, but incompatible old baseline comparisons fail clearly.

## Task 5: establish target hardware and quantum feasibility

**Files:**

- Extend `docs/performance/baseline-protocol.md`.
- Create `docs/performance/quantum-feasibility.md`.
- Create `benchmarks/benchmark_quantum_projection.py` for isolated projection measurements.
- Store actual reports/traces under `benchmark-results/stage-1/`.

**Interfaces:** consumes an available Hopper-or-newer allocation and the reporting helpers; produces a configurable hardware profile and a keep-JAX / native-capability / structured-research decision for required quantum sizes. No fixed model, GPU count, or CPU budget is a library-wide assumption.

- [ ] **Record allocation and topology before benchmarking.** Record GPU architecture/variant, usable memory, topology, CPU affinity, host RAM, driver/runtime, sharing/MIG if present, interconnect, and process mapping. Do not infer usable VRAM or peer links from a model name. Profile one GPU, then two equivalent GPUs if available; additional counts are explicit experiments. Derive worker/thread budgets from actual allocated resources. The earlier 16-core allocation is only an example profile.
- [ ] **Separate support from optimization.** Record validated, experimental, and untested architecture/runtime combinations. Pallas Mosaic is now a candidate for the target GPU family, but each selected kernel still requires dtype/architecture tests and a JAX fallback. Do not fabricate coverage for unavailable successor GPUs.

- [ ] **Use a bounded initial workload ladder.**

| Family | Initial sizes | Accuracy and capacity policy |
|---|---|---|
| CPU classical smoke | 16, 64 | Float64 reference; no GPU performance inference |
| GPU dense classical | 100, 600, 1000, 5000 | Positive and zero-support cases; epsilon 0.5 first, then 0.05 if convergent |
| CPU quantum | subsystem dimensions 2, 4 | Complex reference and partial-trace checks |
| GPU quantum | subsystem dimensions 4, 8, 16, 32 | Complex64 and complex128 with independent tolerance profiles |
| Historical flow | 100 and 600 cells | Accuracy audit before interpreting timing |

The existing benchmark CLI supplies the baseline cases; add case parameters explicitly to its report before varying epsilon or input structure. Larger sizes are explored only after a conservative memory estimate and a smaller run establish headroom. Record the reason for any excluded size.

- [ ] **Budget quantum memory using the actual state.** For `N = n*m`, one dense complex128 matrix uses `16*N*N` bytes. Budget cost, coupling, three correction matrices, projection intermediates, eigenvectors, and eigensolver workspace separately. For equal subsystem dimension 128, one such matrix is 4 GiB; it is not the full solve's peak. Do not assume two GPUs simply pool this memory without a distributed implementation.

- [ ] **Measure `eigh`, full PSD projection, and full solve separately.** In the new benchmark script reuse the reporting helper, enable the selected precision before array creation, generate a Hermitian matrix from a recorded seed, and measure:

```python
def project_psd(matrix):
    import jax.numpy as jnp

    hermitian = (matrix + matrix.conj().T) / 2
    values, vectors = jnp.linalg.eigh(hermitian)
    projected = (vectors * jnp.maximum(values, 0)[None, :]) @ vectors.conj().T
    return (projected + projected.conj().T) / 2
```

Compile the projection, synchronize all timed results, and verify its Hermiticity, eigenvalue clipping, and agreement with a small independent CPU reference. Report workspace/peak memory when measurable. Profiles at this stage describe the existing implementation; no custom kernel is required to collect them.

- [ ] **Assess the native distributed candidate without assuming integration is trivial.** Inspect the available cuSOLVERMp release and target datatypes, column-major/block-cyclic layout, workspace, communicator/runtime dependencies, and the cost of moving between solver and eigensolver layouts. Record whether a GPU-resident FFI operation or a coarse-grained native solver is appropriate. A capability blocker can open a bounded native feasibility task immediately; it need not wait for all classical stages.

- [ ] **Write the decision record.** Include required problem sizes when supplied, measured limits, dominant runtime fractions, feasibility risks, a proposed backend boundary, and the next smallest resolving experiment. If target hardware is unavailable, record that dependency explicitly and allow only hardware-independent stages to proceed.

## Task 6: review the evidence and hand off Stage 2

**Files:** update the Stage 1 evidence documents and roadmap status; do not mark later implementation tasks complete.

- [ ] Run the existing suite plus the new numerical and reporting tests in the baseline environment. Keep optional dependency skips and historical audit failures visible.
- [ ] Verify that no solver formulas, source histories, SDPLab files, or public interfaces changed during evidence collection.
- [ ] Check benchmark JSON schema, case identity, raw samples, outcomes, and the distinction between measured peak memory and estimates.
- [ ] Map each reproduced discrepancy to Stage 2, 3F, or 3Q and each capability blocker to Stage 6 or the early native branch.
- [ ] Refine the Stage 2 core plan using the observed environment and mathematical contracts. Identify which QOT findings belong to separate experimental work and which are prerequisites for a supported entropy solver.

Stage 1 is complete when the numerical contract and evidence are reproducible and unresolved findings have explicit owners. It does not mean that legacy algorithms have been corrected, that GPU performance has passed without hardware, or that the integrated library is ready to release.
