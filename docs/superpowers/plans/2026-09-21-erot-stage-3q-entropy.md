# Stage 3Q quantum entropy implementation plan

> Use superpowers:executing-plans inline. The user has authorized the entire
> roadmap and local commits; no further stage approval is needed.

**Goal:** integrate unequal-dimension quantum operators and a dense entropy-QOT
dual solver with independently verified recovery and explicit failure semantics.

**Architecture:** pure JAX bipartite operators and a trace-one log-partition
dual, followed by a host API adapter. Preserve quadratic compatibility.
**Tech stack:** JAX/NumPy base; SciPy/CVXPY only for independent tests.
**Spec:** ../specs/2026-09-21-qotlib-adoption.md (Q1/Q2).

## Global constraints

Preserve historical checkouts and SDPLab. Attribute QOTLib mathematical/source
adaptations at revision bd534c61aeae082892b9b2421db153beb8e5c804 under
user-confirmed Apache-2.0. No Optax/Flax dependency, implicit placement, global
precision change or file I/O in device solvers. No claimed GPU performance.
Require trace-one positive-definite marginals for this finite-dual solver;
rank-deficient quadratic problems remain supported by Dykstra. Document this
entropy support restriction, rather than silently perturbing input states.

## Review focus

- Near-boundary spectra: report residual failure rather than certify trace alone.
- Complex inputs and unequal dimensions: Hilbert-Schmidt adjoint and Hermiticity.
- Large additive identity shifts: stable spectral recovery and objective constants.
- Zero/one iteration budgets, resume counters and invalid/nonfinite controls.
- Mixed real/complex precision and invalid host method combinations.

## Task 1: bipartite operators and spectral recovery

Files: src/erot/operators/{__init__,quantum}.py,
tests/unit/test_quantum_operators.py, NOTICE, provenance documentation.
Produces partial_traces(matrix, dimensions)->(a,b),
partial_trace_adjoint((u,v))->matrix, and
gibbs_state(matrix, epsilon)->(density, log_partition).
Tensor order is (a,b,a',b'). Recovery uses shifted eigenvalues and softmax;
log_partition is log Tr exp(matrix/epsilon). Dimensions are static.

- [x] Write independent NumPy index-loop trace tests for complex 2x3 systems,
  adjoint identity real(vdot(X,A*(u,v)))=sum real(vdot(A(X),u,v)),
  pure product state traces, shape failures and stable shifted Gibbs tests.
- [x] Run tests/unit/test_quantum_operators.py; expected missing-module failure.
- [x] Implement traces using reshape/einsum and adjoint using Kronecker sums.
  For eigenvalues w and eigenvectors q, use
  p=softmax((w-max(w))/epsilon); density=(q*p)@q.conj().T;
  logZ=max(w)/epsilon+logsumexp((w-max(w))/epsilon).
- [x] Run focused tests; expected all pass. Record provenance and commit.

## Task 2: pure entropy dual solver

Files: src/erot/solvers/quantum_entropy.py, quantum entropy unit/reference tests.
Consumes Task 1 operators. Produces solve_quantum_entropy(cost,a,b,epsilon,
tolerance,max_iterations,*,state=None,learning_rate=1), returning complete
NamedTuple state (dual_a,dual_b,coupling,iterations,step_size) and diagnostics
(primal,dual,gap,feasibility,error,iterations,status).

Primal is Tr(C G)+epsilon Tr(G(log G-I)).
For Y=A*(u,v)-C the trace-one dual is
Tr(a u)+Tr(b v)-epsilon*(log Tr exp(Y/epsilon)+1).
Gradient is (a-Tr_b G,b-Tr_a G); Hermitian traceless gauge fixes both duals.
Use gradient ascent with Armijo backtracking (30 trials); unsuccessful search
returns explicit failure, without accepting its trial. Retain accepted step.
Convergence requires finite values, marginal residual and primal-dual agreement.
No history allocation. Budget is additional work, including zero; checked int32
counters. Exact resume requires identical inputs/controls.

- [x] Write diagonal reference reduced to independently solved classical
  entropic transport; zero-cost product optimum including entropy constant;
  complex noncommuting reference via SciPy optimization with separately coded
  Kronecker sums/eigh; finite-difference complex Hermitian directional derivative.
- [x] Add invalid/nonfinite, singular-marginal, zero/one budget, JIT, batched
  and exact-resume tests. Run before implementation; expected missing solver.
- [x] Implement the stated state machine with fixed-shape device loops and
  status codes compatible with the shared core; line search failure explicit.
- [x] Run focused reference/unit tests; expected all pass. Commit.

## Task 3: host integration, example and scientific contracts

Files: src/erot/api.py, src/erot/cli.py, src/erot/types.py if needed,
examples/quantum_entropy.py, docs/numerics/quantum-entropy.md and API tests.
Host combination: problem='quantum', regularizer='von_neumann', method='dual'.
Keep existing combinations and result fields. Invalid core status must not
appear converged in the host result.

- [x] Add host/CLI invocation tests and unsupported-combination failures.
  Run before adapter; expected new combination rejected.
- [x] Add adapter and runnable complex example with explicit epsilon and
  feasibility reporting. Document objective constants, support restriction,
  full dense eigensolve/memory cost, dtype and resume contracts.
- [x] Run host tests and example; expected success with residual within tolerance.
  Update QOTLib attribution/assessment to distinguish historical audit from
  accepted implementation. Commit.

## Task 4: release checks and final review

- [x] Run full CPU pytest, Ruff check/format, built artifact checks and installed
  base-only entropy API smoke outside checkout; expected green.
- [x] Request one fresh final review under executing-plans; reproduce and fix
  material findings with regressions and green suite.
- [x] Record actual evidence and unexecuted GPU gates; mark roadmap Stage 3Q
  complete only when its requirements hold. Commit and continue Stage 4.
