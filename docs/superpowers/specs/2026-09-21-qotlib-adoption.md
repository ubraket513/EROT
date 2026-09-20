# QOTLib assessment and adoption plan

License update (2026-09-21): the user confirmed Apache-2.0 for both QOTLib and numerical-gradient-flows. Earlier missing-license observations below describe the inspected trees, not an outstanding permission question. Preserve attribution and applicable notices when adapting source.

Status: the initial assessment below is preserved as historical inspection evidence. Stage 1 subsequently executed isolated audits, and Stage 3Q implements the Q1 operators and Q2 dense entropy dual adaptation under user-confirmed Apache-2.0. See docs/numerics/qotlib-audit.md and docs/numerics/quantum-entropy.md for current evidence and limitations. Q3/Q4 remain experimental candidates.

This record complements the [architecture](2026-09-21-erot-flows-hpc-design.md), [cleanup plan](../plans/2026-09-21-erot-stage-0-cleanup.md), and [roadmap](../plans/2026-09-21-erot-flows-hpc-roadmap.md). YACHT remains the engineering reference; QOTLib supplies algorithm candidates, not the destination package structure.

## Inspected source and provenance

The local checkout is `QOTLib/QOTLib`, origin `git@github.com:Pavlo3P/QOTLib.git`, revision `bd534c61aeae082892b9b2421db153beb8e5c804`, dated 2026-01-21. Its working tree was clean during inspection. Serena was used for symbol overviews and targeted implementation reads. Context7 returned no matching QOTLib documentation; this assessment therefore relies on the supplied source, not an unrelated package or assumed public documentation.

The tracked tree contains no discovered license, root README, `pyproject.toml`, or setup configuration. Its requirements include a much broader research stack than EROT needs. The organized tests cover backend operations, spaces, linear operators, low-rank algebra, and NumPy Lanczos. This inspection did not establish equivalent end-to-end solver coverage. No dependencies were installed and no QOTLib tests were run.

Before copying source or distributing adaptations, record the applicable reuse permission/license and attribution. Do not assume that EROT's MIT license applies to this checkout. Preserve the original Git history and map each adopted component to a revision and destination. This prerequisite does not prevent repository cleanup, mathematical design, or independently specified reference tests.

## Existing algorithm candidates

Paths below are relative to `QOTLib/QOTLib/qotlib/` and identify inspected source symbols.

| Candidate | Source | What exists | Adoption decision |
|---|---|---|---|
| QOT constraints | `qot/_problem.py:QOTProblem`, `qot/_constraint_op.py:QOTConstraintOp` | Dense multipartite problem, partial traces, Kronecker-sum adjoint | Reuse the mathematical operator contract; preserve unequal local dimensions supported by EROT |
| Spectral regularization | `regularization/regs/{base,entropy,quadratic}.py` | Spectral entropy and quadratic functions, conjugates, and derivatives | Add explicitly named QOT regularizers after checking normalization and domains |
| Dual optimization | `solvers/_optax.py:run_optax_solver` | JAX differentiation with Optax, complex parameters represented by real/imaginary parts | Adapt into a pure device solver with EROT state/status; keep Optax optional if retained |
| Primal recovery | `regularization/_reg_sdp.py:SDPRegularized.primal_from_dual` | Eigendecomposition and spectral derivative, optionally normalized | Derive recovery for each formulation; do not use unconditional trace normalization |
| Structured block PDHG | `solvers/block/_pdhg.py`, `solvers/block_qot/{pdhg,problem}.py` | Block primal/dual iterations and PSD projections | Experimental workstream after objective, adjoint, step-size, and residual validation |
| Chordal structure | `solvers/block_qot/{chordal,sparse_pattern,submatrix}.py` | Host graph preprocessing and structured tensor-product submatrices | Adopt only with a specified sparsity/completion model and equivalence evidence |
| Matrix-free eigenpair | `linalg/_lanczos.py:stochastic_lanczos` | Matrix-vector iteration for a smallest eigenpair, full stored basis and reorthogonalization | Useful for diagnostics or separately designed algorithms; not a full PSD projection |
| Independent small reference | `solvers/cvxpy/_qot.py:solve_qot_dual` | Unregularized QOT dual, default MOSEK, primal from cone multipliers | Inform optional reference tooling; validate complex conventions and avoid mandatory commercial dependencies |

This is already substantially JAX/NumPy code. A wholesale language translation is unnecessary. There is no basis from the inspected solver paths to advertise an existing production quantum Sinkhorn implementation; spectral dual optimization and block PDHG should be named as such.

## Changes that require numerical validation

1. **Entropy convention.** `EntropyReg` uses `phi(x)=x(log(x)-1)` and a sum-of-exponentials conjugate. `EntropyRegLog` replaces that conjugate with log-sum-exp. Derive the trace-one domain and additive constants for the chosen model; compare primal and dual values under that exact convention. Merely renaming the classes would hide a scientific difference.
2. **Primal feasibility.** `primal_from_dual(normalized=True)` scales spectral weights to sum to one. Trace one alone does not enforce prescribed partial traces, and normalization is not generally the unconstrained spectral derivative for quadratic regularization. Report marginal residuals separately from optimizer gradient norms.
3. **Dimension contract.** `QOTProblem(d, N)` constructs matrices of dimension `d**N`. Retain arbitrary bipartite dimensions in EROT; a future multipartite API should use a tuple of subsystem dimensions. Check partial-trace ordering and the Hilbert-Schmidt adjoint on real and complex inputs.
4. **Dense allocations.** `dual_constr_eig_decomp(..., k=...)` constructs the dense dual slack and calls its full eigendecomposition without using `k`. Primal recovery calls `LowRankMatrix.to_dense()`. A low-rank helper or parameter name does not establish memory scalability.
5. **Optimizer state and stopping.** The Optax driver returns only part of its internal state, allocates histories of length `max_iter`, forces at least two loop iterations, and infers success from exiting before the iteration cap. Specify bounded iteration semantics, finite checks, last-iterate residuals, complete resume state, and configurable logging. Benchmark timing must synchronize device results.
6. **Block PDHG formulation.** One QOT block update hard-codes `l2_reg=1e-4`; the generic block solver computes a regularization contribution that is omitted from its logged objective. Fixed-length loops and periodic debug prints do not provide an optimality certificate. Derive and expose the actual regularized objective and compatible stopping conditions before comparing algorithms.
7. **Chordal equivalence.** A graph completion and small PSD blocks alone do not establish equivalence to the original dense cone. Specify completion variables, overlap consistency, and primal/dual recovery. Test against the dense formulation; label approximations explicitly.
8. **Lanczos limits.** The routine stores a full Krylov basis and diagonalizes a padded tridiagonal matrix. Add an independently computed residual, tests over operator scales and degenerate/complex spectra, and measured memory. A single extremal eigenpair cannot reconstruct all positive eigenvalues needed by dense Dykstra.

These are source-level findings and validation requirements, not claims that every candidate fails on numerical tests.

## Adoption sequence

| Work package | Prerequisites | Proposed destination | Required exit evidence |
|---|---|---|---|
| Q0: provenance and characterization | Stage 0 inventory; source reuse resolved before copying | `docs/architecture/source-provenance.md`, `docs/numerics/qotlib-audit.md`, optional audit tools | Pinned source, dependency isolation, documented objective/normalization and reference cases |
| Q1: operators and spectral primitives | Stage 1 mathematical contract | `src/erot/operators/quantum.py`, focused tests | Partial traces/adjoints, unequal dimensions, complex/rank-deficient cases, agreement with independent dense algebra |
| Q2: dense entropy dual solver | Stage 2 device/result contract; Q1 | `src/erot/solvers/quantum_entropy.py`, `tests/reference/`, quantum examples | Small diagonal analytical case, independent dense reference, primal/dual/feasibility checks, precision and failure tests |
| Q3: structured block PDHG | Q1; explicit block formulation and norm bound | `src/erot/experimental/quantum/`, structured benchmark cases | Dense equivalence where claimed, regularized objective, stopping evidence, memory/time gain |
| Q4: matrix-free spectral work | A specified algorithm that actually needs extremal eigenpairs | `src/erot/experimental/quantum/` initially | Residual-certified eigenpairs and end-to-end solver/approximation evidence; no hidden dense slack |

Q1 and Q2 form roadmap Stage 3Q. Q3 and Q4 are independent research increments, not dependencies of the first gradient-flow release. Do not create all destination modules during cleanup; add them with their validated implementation.

For Q2, start from trace-one density matrices and explicitly state any support restrictions. Use a commuting/diagonal case reducible to independently solved classical entropic OT, then small noncommuting complex marginals with a separately implemented reference. Validate objective constants, stationarity, partial traces, Hermiticity, PSD, nonfinite detection, and termination at the iteration cap. Treat rank-deficient boundary optima carefully rather than requiring a finite dual maximizer in all cases.

The quadratic dual path can later cross-check EROT's Dykstra reference under the same objective. An unregularized SDP oracle is not a valid equality oracle for a finite-epsilon entropy or quadratic solve.

## Integration and hardware boundaries

Expose algorithms through EROT's host problem/result conventions and pure array/PyTree core. Keep numerical iterations free of file I/O, global backend switching, implicit placement, and unconditional printing. Use the YACHT-derived formatting, public type annotations, mathematical docstrings, explicit optional dependencies, and isolated tests. Preserve contributor attribution when adapting permitted source.

Keep NumPy as an independent reference tool; do not import QOTLib's entire backend abstraction merely because it already exists. Avoid coupling EROT to generic SDP drivers or SDPLab. No nested repository becomes a runtime import or mandatory Git submodule.

Hopper-or-newer hardware broadens custom-kernel choices but does not remove exponential multipartite dimension growth, dense spectral memory, or the need for distributed eigensolver design. Profile each accepted algorithm on available hardware and record the precision, usable memory, topology, runtime, and accuracy. Algorithmic structure, JAX compilation, Pallas kernels, CUDA/C++ libraries, and existing Fortran implementations remain separate measurable choices.
