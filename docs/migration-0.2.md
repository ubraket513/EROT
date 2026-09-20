# Migrating to integrated EROT 0.2

EROT now contains the selected gradient-flow and quantum-entropy capabilities.
Install one package; neither the historical numerical-gradient-flows checkout
nor QOTLib is a runtime dependency. SDPLab stays independent. Original source
checkouts and histories are preserved, and no remote repository was moved,
merged, deleted or published by this migration.

Existing `import erot`, `erot.solve`, `SolverConfig`, result fields, CLI defaults
and `.npz` result archives remain available. The `src/erot` layout requires an
installation (`python -m pip install -e .`) instead of relying on the checkout
root as an import path. The [0.1 migration guide](migration-0.1.md) still covers
the earlier `eot`/PyTorch artifact transition. This release adds explicit state
interfaces without making old coupling archives into resumable checkpoints.

## Gradient-flow mapping

| Historical component | Integrated EROT interface | Migration action |
|---|---|---|
| `jko_lab/pdhg_jko.py` | `erot.flows.solve_pdhg_jko` | Pass dense cost, previous cell masses, an energy and numerical controls; consume full state and diagnostics |
| `jko_lab/sinkhorn_jko.py` | `erot.flows.solve_entropic_jko` | Choose the finite-epsilon objective explicitly; pass complete continuation state for an identical subproblem |
| Historical energy helpers | `Entropy(volumes, weight)`, `Quadratic(target, weight)`, `Potential(values)` | Convert sampled density to cell mass and make volumes/weights explicit |
| Historical Lambert-W entropy prox | Energy `.prox` using `erot.optim` | Use stable log-coordinate entropy roots; do not copy the overflow-prone expression |
| Driver time loop/history | `initialize_flow`, `jko_step`, `run_flow_chunk` | Advance only accepted steps; choose static chunk/snapshot sizes and inspect failure status |
| Ad-hoc result dumps | `erot-run` configs and runtime checkpoints | Save complete state and identity metadata; keep snapshots separate from restart data |

A minimal step sequence is provided by [examples/quadratic_flow.py](../examples/quadratic_flow.py):

```bash
python examples/quadratic_flow.py
python examples/heat_flow.py --cells 16 --steps 2 --output results/heat.json
```

These are scientific corrections, not bit-for-bit reproductions of every
historical trajectory:

- Inputs are cell masses, `rho = density * volume`. Entropy is
  `sum rho*(log(rho/volume)-1)` with its continuous zero limit.
- PDHG solves `dt*F(rho) + <C,pi>/2` with both marginal constraints and
  nonnegativity. Defaults use the norm of the coupled density/coupling operator,
  retain extrapolation state, and check feasibility plus stationarity.
- Entropic JKO solves `F(rho) + T_epsilon(rho,previous)/(2*dt)`. Its gradient
  contains the cost-unit potential `f/(2*dt)`, correcting the historical factor.
  Inner transport convergence alone does not establish outer convergence.
- A failed inner solve/line search retains the prior physical density and time.
  Iteration budgets are additional work for an identical resumed subproblem;
  a new physical step starts compatible new subproblem state.
- A blocked point-cloud inner solve reduces stored geometry/plan memory; it
  preserves the finite-epsilon objective and quadratic arithmetic. PDHG still
  stores the dense coupling and is not made distributed by selecting geometry.

See [full conventions](numerics/gradient-flows.md), the
[historical flow audit](numerics/legacy-flow-audit.md), and the
[heat refinement study](numerics/heat-flow-study.md). Finite-epsilon JKO and
unregularized JKO have different objectives and reference solutions.

## QOTLib mapping

| QOTLib source contract | EROT adoption | Difference / limit |
|---|---|---|
| `qotlib/qot/_constraint_op.py` | `erot.operators.quantum` partial traces and adjoints | Explicit `(a,b,a',b')` tensor order, complex Hilbert-Schmidt adjoint tests, unequal dimensions |
| `regularization/_reg_sdp.py`, `regs/entropy.py` | Shifted Gibbs recovery and `erot.solvers.quantum_entropy` | Trace-one entropy conjugate includes the required constant; marginal/primal-dual checks replace trace-only acceptance |
| Dual optimizer driver | Bounded pure JAX ascent/backtracking and `QuantumEntropyState` | No mandatory Flax/Optax, no unbounded history; complete continuation and failure diagnostics |
| Backend/space abstractions, block/chordal/Lanczos candidates | Not imported | Need separately stated formulations and equivalence/error/performance evidence |

Use the public wrapper with `problem="quantum"`,
`regularizer="von_neumann"`, `method="dual"`, or run:

```bash
python examples/quantum_entropy.py
```

The dense entropy path requires positive-definite trace-one marginals. It does
not silently perturb rank-deficient inputs. The existing quadratic/cyclic
quantum solver remains available for rank-deficient marginals. Entropy recovery
uses the full joint spectrum; there is no distributed quantum eigensolver,
quantum gradient flow or matrix-free entropy guarantee. See
[entropy conventions](numerics/quantum-entropy.md) and the
[QOTLib audit](numerics/qotlib-audit.md).

## Provenance and reproducibility

The pinned flow source is `0c6f5b50aee9ab5378970e28d700ab8fd193f554` (Dohyoung Ko);
QOTLib is `bd534c61aeae082892b9b2421db153beb8e5c804` (Pavlo Pelikh).
The user confirmed Apache-2.0 for both source projects. Adapted components carry
Apache identifiers and attribution in [NOTICE](../NOTICE); the original EROT
MIT terms remain in [LICENSE.txt](../LICENSE.txt). Distribution metadata records
`MIT AND Apache-2.0`, with both license texts. See
[source provenance](architecture/source-provenance.md) for paths and changes.
YACHT inspired packaging/testing/style choices; no YACHT code was copied.

Historical checkouts, archives and generated outputs are excluded from release
artifacts. The retained `erot.experimental.classical` module remains available
but experimental. No unaccounted source deletion is needed for this release.
Use [experiment configurations](../experiments/configs),
[restart guidance](numerics/experiments.md), and the
[support matrix](support-matrix.md) to choose a validated execution path.
Checkpoint metadata is deliberately strict: legacy snapshots, changed source,
changed inputs or changed distributed topology are not accepted as exact resume.
