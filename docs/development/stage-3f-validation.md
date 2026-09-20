# Stage 3F verification

Gradient flows now use the shared transport core with explicit accepted physical
time, complete backend continuation state and selective snapshots. Independent
CVXPY references cover both unregularized PDHG JKO and finite-epsilon JKO with
entropy and quadratic energies. Scalar entropy proximal roots are checked
against SciPy. These are different discrete models, with their conventions in
[gradient flows](../numerics/gradient-flows.md).

Commits `01389da`, `0dedc28`, `b608e3b` and `fc26107` introduce the energies,
PDHG, entropic optimization and trajectories. The
[heat study](../numerics/heat-flow-study.md) records spatial, time-step,
tolerance and epsilon changes against a nonstationary Neumann heat solution.
It exposes coarse-grid pinning and regularization bias; numerical convergence
of an optimizer is not presented as continuum convergence.

Fresh review of `5add444..fc26107` identified mixed floating precision breaking
compiled loop carry types. A regression reproduced this for both backends;
state and snapshots now use a common input precision. Two entropy findings
were also treated as material: valid extreme mass/volume ratios could overflow
before taking a logarithm, and zero-weight entropy differentiation at zero mass
returned NaN. All four regression cases failed before their fixes and pass now.
The final full CPU suite reports **125 passed, 5 skipped** (three unavailable
GPU checks and two opt-in artifact checks). Ruff passes all 57 Python files.

Rebuilt wheel and sdist content checks pass separately (two tests), including
the Apache-2.0 license and provenance NOTICE. A base-only wheel installation
outside the checkout passes both mixed-precision flow backends. An additional
three-cell smoke with a poorly conditioned transport kernel exhausted its
inner budget and correctly returned failure; successful smoke verification
uses the documented two-cell regression fixture and epsilon 0.2. No solver
tolerance was relaxed to turn that failed solve into a success.

No GPU throughput or hardware capacity is certified. Dense flow steps still
materialize couplings; memory scaling follows in Stage 4. The original flow
checkout, QOTLib checkout and SDPLab remain unchanged. Alternative optimizers
are optional future extensions, not base dependencies. The library supports
JAX mirror and projected SGD steps without Flax or Optax.
