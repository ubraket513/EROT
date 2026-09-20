# Discrete gradient-flow conventions

Flow inputs are cell masses, not density samples. Convert densities using
`mass = density * cell_volume`. `Entropy(volumes, weight)` represents
`weight * sum mass*(log(mass/volumes)-1)`, including the continuous zero limit.
`Quadratic(target, weight)` represents `weight/2 * ||mass-target||²`, and
`Potential(values)` represents linear potential energy. All have a nonnegative
domain and value, gradient and Euclidean proximal maps. They are JAX PyTrees;
no Flax/Optax dependency is required.

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
requires a new subproblem. This pure solver does not yet advance physical time;
the trajectory layer will accept a density only after a successful subproblem.

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
