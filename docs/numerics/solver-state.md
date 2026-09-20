# Composing and resuming Sinkhorn solves

`erot.solve` retains its validated, synchronized dense result. For compiled
numerical code, import `solve_sinkhorn`, `materialize_plan` and
`SinkhornWarmStart` from `erot.solvers`:

```python
import jax
import jax.numpy as jnp
from erot.solvers import solve_sinkhorn, materialize_plan, SinkhornWarmStart

cost = jnp.array([[0., 1.], [1., 0.]])
marginals = (jnp.array([.4, .6]), jnp.array([.5, .5]))
run = jax.jit(solve_sinkhorn)
state, diagnostics = run(cost, marginals, .5, 1e-6, 1000)
plan = materialize_plan(cost, state.potentials, .5)
```

The pure function does not choose devices, enable precision globally, convert
arrays to NumPy, synchronize or return Python scalar diagnostics. Configure
precision and place inputs before calling it. Shapes/rank and presence of
optional state determine structure; epsilon, tolerance and iteration budget are
dynamic. The core supports `jit`, `vmap` and `scan`. It does not claim reverse-mode
differentiation through a dynamically terminated loop.

Status codes in `erot.solvers.state` are `CONVERGED=0`, `ITERATION_LIMIT=1`,
`INVALID_INPUT=2`, and `NUMERICAL_FAILURE=3`. Check status before using a result.
Invalid input may retain nonfinite diagnostics; it is never marked converged.
Static shape/type mismatches raise errors during tracing. Marginal error is the
maximum marginal L1 residual. No plan is stored in the returned state, although
the dense implementation currently forms dense intermediate tensors.

Potentials use cost units: the plan is `exp((sum(f_k) - C)/epsilon)`. After a
completed sweep the first k-1 potentials have marginal-weighted mean zero and
the last compensates their shifts. Inactive support uses minus infinity and
contributes no mass. The marginal derivative is the potential, not twice the
potential, under the documented Shannon convention.

Pass `state=state` to continue the **identical** cost, marginals, epsilon and
dtype. The iteration budget is additional sweeps; the returned count is
cumulative. Zero additional work is valid. The low-level function cannot verify
source identity without retaining the whole problem; callers must enforce this
contract. Checkpoint manifests will validate identity at the host boundary.

For a related problem, pass `warm_start=SinkhornWarmStart(state.potentials)`
instead. The count resets; cost-unit potentials are interpreted at the new
epsilon. Previously inactive support that becomes positive initializes to zero.
Warm starts reach the same accuracy but need not save work on every problem.
The regression suite records a representative nearby-marginal case that does.

The host API now preserves a single device shared by explicitly supplied JAX
arrays when `device='auto'`. Ambiguous multi-device placement requires an explicit
device or the pure API. Host validation still deliberately inspects inputs;
use the pure API inside compiled flows. Nonfinite controls and fractional
iteration caps are rejected instead of silently truncating or failing later.

## Quantum quadratic resume

`erot.solvers.quantum.solve_quantum_quadratic` returns a
`QuantumDykstraState` containing the coupling, all three Dykstra corrections
and cumulative sweep count, plus array-valued diagnostics. Pass it back as
`state=` with identical cost, marginals, epsilon and dtype. A coupling snapshot
alone is not sufficient; corrections cannot be reused across a changed problem.
The pure function retains complex marginal information even for real costs.
At zero work it reports the initial marginal and PSD infeasibility. Initial
validation and restart diagnostics require additional small/dense spectral
checks; benchmark this overhead separately from the iterative projection loop.
The existing quantum dense wrapper uses this core without changing its result
signature or quadratic objective.

Iteration budgets and cumulative counts must fit signed int32; overflow is an
invalid input, never wrapped work. Classical complex marginals are rejected
before any dtype conversion. The legacy three-array solver wrappers represent
invalid-input/numerical-failure status with infinite error so a finite initial
residual cannot be mistaken for convergence.
