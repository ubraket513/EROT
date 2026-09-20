# Stage 3F: gradient flows on the shared OT core

Execute inline; the user has authorized the full roadmap and important local
commits. Stage 2 supplies cost-unit potentials and pure state. Preserve the
legacy checkouts and their Apache-2.0 provenance; document adaptations.

## Task 1: energies and stable proximal operators

- [x] Implement callable JAX-compatible entropy, linear potential and quadratic
  energy objects with value, gradient and Euclidean prox. Use cell masses;
  entropy is sum rho*(log(rho/volume)-1), with 0 log 0 = 0. Volumes must be
  positive. Quadratic energy is weight/2 * ||rho-target||².
- [x] Implement entropy prox through a stable scalar equation in log space,
  avoiding exp(z/alpha) overflow. Handle alpha zero by projection onto the
  nonnegative domain and return explicit invalid controls from device solvers.
- [x] Test scalar optimality against independent SciPy roots for large positive,
  negative and small-alpha inputs, plus prox/gradient/objective conventions.
- [x] Keep base dependencies JAX and NumPy; no Flax or Optax required.

## Task 2: one correct unregularized PDHG JKO step

- [x] Define full PDHG state (coupling, density, both duals, extrapolated
  iterates and work count). Solve dt*F(rho)+0.5*<C,pi>, pi>=0,
  rows(pi)=rho, cols(pi)=previous. Honor theta and both step sizes.
- [x] Use the actual coupled constraint operator K(pi,rho)=(rows(pi)-rho,
  cols(pi)); prove/bound its norm, and require tau*sigma*||K||²<1.
- [x] Convergence includes both equality residuals, positivity, finite values
  and primal/dual stationarity; fixed work alone is not success. Return raw
  feasible iterates, never conceal mass errors by normalizing outputs.
- [x] Compare quadratic and entropy JKO cases to independently stated CVXPY
  problems, including rectangular costs where supported and invalid steps.

## Task 3: entropic JKO from shared Sinkhorn potentials

- [x] Implement constrained density optimization of F(rho)+T_epsilon(rho,prev)
  /(2*dt). The transport derivative is f/(2*dt), fixing the audited factor two.
- [x] Use a documented projected or mirror gradient algorithm with explicit
  step-size/backtracking policy and simplex/mass constraints. Retain a plain
  JAX gradient option; any optimizer variants stay optional.
- [x] Reuse cost-unit potentials between outer iterations. An unsuccessful OT
  solve fails the outer step; marginal feasibility alone cannot certify the
  density minimizer. Return objective and constrained-stationarity diagnostic.
- [x] Match the finite-epsilon objective to a joint-coupling CVXPY reference.
  Test epsilon scaling separately from the unregularized PDHG model.

## Task 4: trajectories, failure semantics and scientific examples

- [x] Add FlowState with current mass, accepted count, physical time and backend
  continuation state. Separate snapshots from full solver checkpoints.
- [x] Provide jko_step and a fixed-length chunk runner with explicit snapshot
  selection. On failure retain previous physical state/time; expose attempted
  work and reason. No hidden adaptive change of the modeled objective.
- [x] Test interruption-ready state structure, no time advance on inner failure,
  mass conservation, entropy/quadratic reference trajectories and scan use.
- [x] Add runnable heat-flow and quadratic-flow examples. State grid, boundary
  conditions, cell volumes and independent space/time/epsilon/tolerance
  refinement parameters. A heat study must compare evolution to a known
  solution, not only equilibrium.

## Task 5: integration and review

- [x] Update provenance, public API guidance and numerical limitations.
- [x] Run full CPU suite, new reference tests, CLI/install checks as needed and
  Ruff. Prepare GPU checks without claiming unavailable hardware results.
- [x] Commit important increments; request the execution skill's final review,
  fix material findings and continue to Stage 3Q without asking permission.

Acceptance: one entropic and one unregularized model match their own independent
references; entropy and quadratic energies work; pure JAX composition and
explicit failure/time semantics are demonstrated. Large-system memory scaling
belongs to Stage 4, not this dense integration milestone.
