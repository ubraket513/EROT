# Discrete gradient-flow conventions

Flow inputs are cell masses, not density samples. Convert densities using
`mass = density * cell_volume`. `Entropy(volumes, weight)` represents
`weight * sum mass*(log(mass/volumes)-1)`, including the continuous zero limit.
`Quadratic(target, weight)` represents `weight/2 * ||mass-target||²`, and
`Potential(values)` represents linear potential energy. All have a nonnegative
domain and value, gradient and Euclidean proximal maps. They are JAX PyTrees;
no Flax/Optax dependency is required.

Solvers promote floating state to the common dtype of cost, masses and array
energy parameters before entering compiled loops. This includes trajectory
snapshots and continuation state. Enable JAX x64 explicitly when float64 is
required; promotion does not change the global precision configuration.
Entropy evaluates logarithms separately to avoid overflowing mass/volume
ratios. Zero-weight entropy has zero derivative, including at zero mass.

## Unregularized PDHG JKO

```python
import jax
import jax.numpy as jnp
from erot.flows.functionals import Entropy
from erot.flows.pdhg import solve_pdhg_jko

jax.config.update("jax_enable_x64", True)
x = jnp.linspace(0., 1., 4)
cost = (x[:, None] - x[None, :]) ** 2
previous = jnp.array([.1, .5, .3, .1])
energy = Entropy(jnp.full(4, .25))
state, diagnostics = jax.jit(solve_pdhg_jko)(
    cost, previous, energy, .4, 1e-8, 30000
)
```

This solves `dt*F(rho) + 0.5*sum(C*pi)` with nonnegative coupling and density,
`rows(pi)=rho` and `columns(pi)=previous`. Finite-epsilon Sinkhorn JKO has a
different objective and requires its own reference.

For rectangular n-by-m cost, the coupled constraint operator acts on both
coupling and density:

```text
K(pi,rho) = (rows(pi)-rho, columns(pi))
||K||² = (m+n+1 + sqrt((m+1-n)²+4*n*m))/2.
```

Default primal and dual steps are `0.99/||K||`. Explicit steps must satisfy
`primal_step*dual_step*||K||² < 1`; theta is explicit in [0,1]. The default
theta=1 is standard Chambolle–Pock extrapolation. No convergence-rate guarantee
is asserted for every other allowed setting. The implementation uses dual-first
updates and retains extrapolated primal state for exact continuation.

Success requires feasibility and stationarity at the returned iterate.
Feasibility is the maximum absolute row/column residual and negative-entry
violation. Stationarity is the maximum unit-step primal proximal fixed-point
residual for coupling and density. It uses the stated scaled objective; it is
not an unscaled gradient norm. Objective, both diagnostics, combined error,
work count and status are arrays. No post-solve normalization conceals errors.

`PDHGState` includes duals, coupling, density, extrapolated coupling/density and
cumulative work. Resume only the identical subproblem and step parameters using
`state=`; budget is additional work. Changing the physical previous density
requires a new subproblem. This pure solver does not advance physical time;
the trajectory layer accepts a density only after a successful subproblem.

Entropy prox solves `x + alpha*log(x/volume) = z` in log coordinates without
forming `exp(z/alpha)`. Alpha zero projects onto the nonnegative domain.
Roots below the dtype's normal range return zero, matching accelerator
underflow. Invalid controls/volumes produce NaNs for solver status handling.
Independent root tests cover alpha zero through 1e4, z from -100 to 1e4 and
nonuniform volumes. More extreme values need problem-specific conditioning checks.

Quadratic and entropy four-cell steps agree with independent CVXPY/CLARABEL
solutions of the same objective. Rectangular resumed runs reproduce fixed
uninterrupted work exactly. These CPU tests do not establish GPU performance,
continuum convergence or a complete heat-equation study.

## Finite-epsilon JKO

`solve_entropic_jko` minimizes `F(rho)+T_epsilon(rho,previous)/(2*dt)`.
Its density gradient uses `energy.gradient(rho)+f/(2*dt)` from the shared
cost-unit potentials, correcting the historical factor of two. The default
mirror step preserves positive mass; `method='sgd'` uses Euclidean simplex
projection. Both use Armijo backtracking (at most 30 trials per update).
Choose method statically under `jit` and set `inner_tolerance <= tolerance`.

The line search uses the transport dual estimate: it is less sensitive to small
marginal infeasibility than the approximate primal objective near stationarity.
The reported objective remains the primal energy plus transport cost/entropy.
Convergence requires successful inner transport, marginal/mass feasibility and
the maximum absolute centered density gradient. Inner OT convergence alone is
insufficient. The solver retains positive densities down to the dtype's normal
range; extremely concentrated solutions may need higher precision or fail the
stationarity criterion rather than silently certifying a boundary approximation.

`EntropicJKOState` retains density, transport potentials, outer count, all inner
sweeps (including rejected line-search trials) and accepted step size. Exact
resume requires unchanged problem inputs and compatible numerical controls.
Counter overflow is a failure, not silently wrapped work. Status 4 denotes an
unsuccessful inner solve; status 5 denotes exhausted backtracking. Existing core
status codes 0–3 retain their meanings. Quadratic and entropy three-cell cases
match independent joint-coupling CVXPY references at finite epsilon.

## Accepted trajectories and snapshots

```python
from erot.flows import initialize_flow, run_flow_chunk

flow = initialize_flow(previous, backend="pdhg")
chunk = run_flow_chunk(
    flow, cost, energy, .4, 1e-8, 30000,
    steps=4, backend="pdhg", snapshot_stride=2
)
```

`FlowState` contains the accepted density/time, accepted and attempted counts,
accumulated solver work, last status and complete backend state. A failed solve
retains the previous physical density and time. An explicit retry through
`jko_step`, or a new chunk, resumes the failed subproblem. Do not change its
cost, energy, previous density or physical time step during that retry; create
a new flow state for a different problem. After success, the next physical step
creates a fresh subproblem instead of reusing incompatible PDHG corrections.

A chunk stops attempting steps after a failure. Only the initial snapshot and
every `snapshot_stride` attempted-position snapshot are allocated; pass `None`
to omit all snapshots. Final continuation state is always returned. Repeated
snapshot times after failure mean unchanged physical state. `steps` and stride
are static under JAX compilation. Snapshots contain masses and times; they do
not replace the full backend state needed for a checkpoint.
