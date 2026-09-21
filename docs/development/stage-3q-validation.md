# Stage 3Q verification

Historical integration record, captured before the subsequent repository cleanup.
See [the current repository map](repository-map.md) for the retained layout;
removed files and detailed task plans remain in EROT Git history.

Q1/Q2 are implemented as dense bipartite operators and a trace-one entropy-QOT
dual solver. Complex unequal-dimension traces, Hilbert-Schmidt adjoints and
rank-one product traces match independent index algebra. Rank-deficient
operator inputs are supported; the finite-dual entropy optimizer explicitly
requires positive-definite trace-one marginals.

The diagonal entropy solution agrees with independently iterated NumPy
classical Sinkhorn. A complex noncommuting problem agrees with a separately
implemented NumPy/SciPy BFGS dual using a Hermitian basis. Tests cover entropy
constants, primal recovery, objective values, complex directional derivatives,
marginal feasibility, exact resume, zero/one caps, JIT, batching and invalid
input. The public API/CLI expose von_neumann/dual while preserving the existing
quadratic default.

Commits fe7d005, 0977c80 and f9ad2a6 establish the operators, solver and host
integration. Fresh review of 16b36a6..f9ad2a6 found two important defects: a
large identity cost shift degraded the optimizer, and mixed-precision floating
budgets could bypass int32 range checks. Both were reproduced. The fix centers
the cost before spectral slacks and line search, restores objective offsets at
reporting, validates each counter before narrowing and checks cumulative work
using integers. Five new regression cases cover shifted evaluation/solve and
rounded budgets/wide counters. No review findings remain deferred.

After the fix, the complete CPU suite reports **150 passed, 7 skipped**: five
GPU cases and two opt-in artifact checks. Ruff checks and formatting pass all
65 files. Rebuilt wheel/sdist inspection passes separately (two tests). The
rebuilt base-only installed wheel passes complex shifted-cost API solves outside
the checkout in float32 and float64. Apache-2.0 notices and licenses are present
in artifacts; historical source checkouts and SDPLab were not changed.

The complex example converges and reports primal, dual, signed gap, marginal
residual and minimum eigenvalue. The signed gap is not advertised as a bound
before feasibility. The solver still performs a full dense eigendecomposition
per line-search trial. No matrix-free, distributed spectral or GPU performance
claim is made. CPU/GPU agreement tests are prepared for both precisions; actual
execution awaits the user's hardware. Q3/Q4 block/chordal and Lanczos paths
remain experimental research candidates, not silently promoted capabilities.
