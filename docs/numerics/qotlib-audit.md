# QOTLib adoption audit

The read-only source is QOTLib revision
`bd534c61aeae082892b9b2421db153beb8e5c804`. The user confirmed Apache-2.0
reuse rights for this checkout; attribution remains required during adaptation.
No QOTLib implementation has been added to the EROT runtime in this stage.

Run the audit in an isolated environment with EROT's reference dependencies,
Optax, and NetworkX installed:

```bash
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu \
  .venv-legacy-audit/bin/python benchmarks/audit_qotlib.py \
  --source-path /path/to/QOTLib \
  --output benchmark-results/stage-1/qotlib-audit.json
```

The supplied path must contain `qotlib/`. The audit suppresses bytecode writes,
records the source revision, and returns nonzero for historical mismatches or
errors. These outcomes are evidence, not failing EROT CI requirements. The
recorded run used Python 3.12, JAX 0.11.2, NumPy 2.5.3 and CPU execution.

## Operator and entropy evidence

Complex Hermitian partial traces for two and three local two-dimensional spaces
agree with independent NumPy tensor contractions to 1.2e-16. Real Hilbert–Schmidt
adjoint identities agree to 9e-16. The source operator supports equal local
dimensions; EROT's unequal bipartite dimensions must remain supported.

For zero cost, two maximally mixed qubit marginals, and epsilon 0.7, the exact
optimal coupling is I/4. The entropy convention
`epsilon * Tr(Gamma (log(Gamma) - I))` gives -1.6704060527839235.
`EntropyReg` reproduces both primal and dual values. `EntropyRegLog` gives
-0.9704060527839233 for its dual, an offset of +epsilon from its inherited
primal objective. For the trace-one domain, the conjugate of this convention is
`epsilon * (log Tr exp(Y/epsilon) + 1)`. Dropping the constant requires changing
the primal convention to `epsilon * Tr(Gamma log(Gamma))` as well.

The complex Hermitian directional derivative of the exponential dual is
-2.091862809863258 by automatic differentiation and -2.091862809949774 by a
central difference with step 1e-5. This check includes imaginary off-diagonal
entries and a nondegenerate test point.

For marginals diag(0.2, 0.8) and diag(0.6, 0.4), recovering the zero-dual
matrix with default normalization produces trace one but marginal residual
0.4472135954999579. This is an intentional demonstration of the limitation:
normalization alone cannot establish convergence or feasibility. The default
normalization also changes the unconstrained exponential conjugate derivative;
a supported solver must choose one formulation explicitly.

## Optimizer and block solver evidence

A zero iteration cap raises `IndexError` from a zero-length history. A one-step
SGD cap returns diagonal dual entries -0.02916096488652795, whereas the exact
single update from zero with learning rate 0.01 is -0.015. The source loop
forces at least two iterations, while its returned history contains one entry.
The wrapper returns a dual and histories, but not a complete optimizer restart
state. Its timer does not synchronize the dispatched result. These are
migration requirements, not timing or restart guarantees.

The block PDHG audit uses one full clique of dimension four, cost
`diag(0, 1, 1, 0)`, maximally mixed marginals, and all primal/dual step sizes
0.2 with extrapolation one. The actual dual problem is

```text
maximize  <marginals, U> - (1e-4 / 2) ||V||_F^2
subject to KroneckerSum(U) + V = C, V >= 0.
```

The regularization is hard-coded in `v_step`; the source objective logger omits
its penalty. An independent CVXPY/CLARABEL formulation gives approximately
-9.9999222e-5. At 1,000 iterations the equality residual is 1.12516e-4, the
reported objective is 6.27098e-5, and the corrected objective is -3.72678e-5.
That iterate is not sufficiently feasible for an objective comparison. The
3,000-iteration run reduces the equality residual to 2.01e-12 and gives a
corrected objective of -0.000100000001089, agreeing with the independent
reference to 7.8e-10. Its bare logged value is approximately zero, demonstrating
why the omitted penalty matters even at a feasible iterate.
Neither a fixed iteration count nor a positive semidefinite slack alone proves
convergence. This audit does not validate arbitrary overlapping clique models.

## Spectral scope and adoption decisions

The source constructs the dense dual slack, performs full `eigh` even when a
rank argument is supplied, and densifies its low-rank primal representation.
These paths do not establish memory scalability. Independent Lanczos checks on
eight-dimensional complex arrays with diagonal spectra at scales 1 and 1e12
recover the minimum eigenvalue with normalized vectors and relative residuals
below 2e-15. Existing real/complex Lanczos tests also pass. These results do not
make a smallest-eigenpair oracle equivalent to full positive-semidefinite
projection, nor establish matrix-free whole-solver execution.

The selected original linear-operator test suite is stale: 18 failures and 43
setup errors originate from the removed `DenseVectorSpace(shape=...)`
constructor argument. Three Lanczos tests pass. Record fixture incompatibility
separately from numerical algorithm failures; the independent operator audit
uses the current constructors.

| Candidate | Decision and owner |
| --- | --- |
| Partial traces, Kronecker-sum adjoint | Stage 3Q/Q1: adapt with unequal-dimension and complex tests |
| Entropy dual and primal recovery | Stage 3Q/Q2: choose consistent constants, test returned residuals and iteration caps, expose restart state |
| Block/chordal PDHG | Experimental Q3: retain actual regularized objective; no stable API claim from this tiny case |
| Lanczos and low-rank representations | Experimental Q4: require whole-solver memory and spectral error contracts |

Nonfinite optimizer termination, arbitrary complex initializations, and restart
reproducibility remain acceptance tests for the new solver. The historical
wrapper supplies insufficient state for a restart comparison. No GPU,
distributed eigensolver, large-dimensional feasibility, or performance result
is claimed by this audit.
