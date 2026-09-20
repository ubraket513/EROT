# Stage 1 evidence and limits

The baseline stage adds reference tests, audit/measurement tools and documents.
It does not change any EROT solver formula. Stage 0 source/artifact provenance
and isolated-install evidence remain applicable; the new runtime source tree is
unchanged from Stage 0.

Local commits: `afd55bf` independent objectives and legacy flow audit;
`6e23e90` matched-accuracy reporting; `c8bc7dd` QOTLib audit; `65a9bb9`
hardware inventory and quantum projection validation.

Python 3.12 CPU verification: **66 passed, 4 skipped**. The skips are two CUDA
checks and two distribution-inspection checks requiring explicit build paths.
Ruff check and format pass on `src tests benchmarks` (36 Python files).
Stage 0 separately verified both built distributions and Python 3.11; these are
not represented as fresh Stage 1 build/minimum-version runs.

Evidence is reproducible from the commands in
[numerical conventions](../numerics/conventions.md),
[flow audit](../numerics/legacy-flow-audit.md),
[QOTLib audit](../numerics/qotlib-audit.md),
[baseline protocol](../performance/baseline-protocol.md), and
[quantum feasibility](../performance/quantum-feasibility.md).
Raw machine reports live under ignored `benchmark-results/stage-1/`.
Historical audit nonzero exits are retained deliberately, not hidden in CI.
The source checkouts remain unmodified.

The user confirmed no target GPU/Slurm access. CPU inventory is available;
explicit GPU inventory records an unavailable JAX GPU backend and exits 1.
Consequently the planned GPU workload ladder, topology/communication
measurements, native layout round trip and performance decisions remain
unexecuted. This is the authorized hardware-independent handoff, not a GPU
performance gate pass. Stage 5/6/7 tooling will extend this preparation.

Discrepancy owners:

- Stage 2: cost-unit potential state, precise budgets and device status.
- Stage 3F: historical factor-of-two gradient, PDHG construction/tracing and
  feasibility, entropy-prox overflow/domain behavior.
- Stage 3Q/Q1-Q2: consistent entropy dual constants, marginal feasibility,
  bounded iteration/state handling, nonfinite and restart acceptance checks.
- Experimental Q3/Q4: block penalty and clique formulation; full spectral versus
  approximate eigenpair/low-rank memory contracts.
- Stage 6/7: distributed memory layout, native eigensolver feasibility and
  measurements on real GPUs.

The separate Stage 2 plan specifies the next implementation contracts. Final
Stage 1 review found two important audit-gate omissions: signed classical plans
and indefinite block slack could pass objective/equality checks. Both were
reproduced before correction. Explicit nonnegativity, Hermiticity and PSD gates
now reject them; regression tests pass. Historical audit conclusions remain
unchanged under the stricter checks. No Critical findings were reported.
