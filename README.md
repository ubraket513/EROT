# EROT

EROT is a JAX-first research library for entropy-regularized classical and
quantum optimal transport. The integrated 0.2 research release provides:

- multi-marginal log-domain Sinkhorn for Shannon-regularized classical OT;
- two-marginal cyclic dual projection for quadratic classical OT;
- Dykstra cyclic projection for quadratic quantum OT with general Hermitian
  density matrices and positive-semidefinite couplings;
- dense von Neumann entropy QOT by dual optimization for positive-definite
  marginals, with explicit primal-dual and marginal diagnostics;
- discrete gradient flows through PDHG and finite-epsilon Sinkhorn JKO;
- blocked point-cloud transport, reproducible experiment workers and checkpoints;
- experimental row-sharded Sinkhorn with distributed launch/validation tools.

CPU numerical and workflow checks are validated. Actual Hopper-or-newer GPU,
Slurm and multi-node performance is **not yet validated**. See the
[support matrix](docs/support-matrix.md) for tested runtimes and capability limits.

See [gradient-flow conventions](docs/numerics/gradient-flows.md) and
[quantum entropy](docs/numerics/quantum-entropy.md) for objectives and limits.

## Installation

CPU-only installation:

```bash
python -m pip install .
```

For NVIDIA GPUs, install the extra matching the machine's driver/toolchain:

```bash
python -m pip install -e '.[cuda13]'
# or: python -m pip install -e '.[cuda12]'
```

JAX's current platform requirements are documented in the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Python API

```python
import numpy as np
import erot

a = np.array([0.4, 0.6])
b = np.array([0.5, 0.5])
C = np.array([[0.0, 1.0], [1.0, 0.0]])

result = erot.solve(
    C,
    [a, b],
    problem="classical",
    regularizer="shannon",
    method="sinkhorn",
    config=erot.SolverConfig(epsilon=0.1, device="auto"),
)

print(result.coupling)
print(result.error, result.iterations, result.converged)
```

Float64 is the default. Passing `dtype="float64"` enables JAX x64 before
input conversion. Use `dtype="float32"` when its accuracy is sufficient; measure speed on your
workload and hardware. Pure kernels require precision to be configured before
constructing inputs.

## CLI

Generate inputs and solve a classical problem:

```bash
erot generate cost --n 100 --output C.npy
erot generate marginal --n 100 --loc 2.1 --output a.npy
erot generate marginal --n 100 --loc -3 --scale 0.75 --output b.npy

erot solve classical \
  --cost C.npy \
  --marginal a.npy \
  --marginal b.npy \
  --regularizer shannon \
  --method sinkhorn \
  --epsilon 1 \
  --output result.npz

# Optional: install EROT[plot] before plotting
erot plot result.npz --output coupling.png
```

Use `erot solve quantum` for general density matrices; quadratic regularization
and cyclic projection are selected automatically. Select entropy QOT with
`--regularizer von_neumann --method dual`. Solver results use `.npz`
and contain the coupling, residual, iteration count, convergence flag, elapsed
time, and JSON metadata. Plotting is optional and requires `EROT[plot]`.

## Gradient flows and experiments

After installing the package, run the checked-in CPU examples:

```bash
python examples/quadratic_flow.py
python examples/heat_flow.py --cells 16 --steps 2 --output results/heat.json
python examples/quantum_entropy.py
erot-run experiments/configs/classical-blocked.json --output-root results
```

Flow inputs are cell masses. PDHG and finite-epsilon Sinkhorn JKO solve different
objectives; check [their conventions](docs/numerics/gradient-flows.md) before
comparing trajectories. See [experiment/restart guidance](docs/numerics/experiments.md),
[configurable HPC launch](hpc/README.md), and
[experimental distributed mode](docs/numerics/distributed.md). The distributed
CPU validation runtime is pinned separately from the base dependency floor.

## Numerical and performance notes

The full iteration loops execute through `jax.lax.while_loop` inside `jax.jit`.
The first solve for a new shape includes compilation; later solves reuse the
compiled executable. Benchmark compilation and warm runtime separately and
call `block_until_ready()` before stopping a GPU timer.

Dense multi-marginal costs require memory proportional to the product of
marginal sizes. The blocked two-cloud Shannon path avoids a dense cost/coupling,
while retaining quadratic arithmetic. PDHG still stores its dense coupling.
Dense quantum couplings require `O((nm)^2)` storage and full spectral work;
classical sharding does not provide a distributed quantum eigensolver.
No reverse-mode differentiation through dynamic convergence loops is promised.

JAX remains the implementation choice. C++/CUDA, Pallas or an existing Fortran
library can be added when measured full-solve benefit or a required capability
justifies them; see the [native decision](docs/performance/native-decision.md).
[Benchmark protocols](docs/performance/baseline-protocol.md) distinguish
compilation, synchronized execution, compiler storage and measured live memory.

See the [0.2 migration guide](docs/migration-0.2.md) for numerical-gradient-flows
and QOTLib mappings, scientific corrections, attribution and retained history.
The [0.1 guide](docs/migration-0.1.md) covers legacy `eot` and PyTorch `.pt` files.

## Development

The package lives under `src/erot/`. For an editable development installation:

```bash
python -m pip install -e '.[test,plot,dev]'
python -m pytest
ruff check src tests benchmarks
ruff format --check src tests benchmarks
```

See [contributing](CONTRIBUTING.md), [distribution checks](docs/development.md),
and [the repository map](docs/development/repository-map.md).
Historical demo outputs and the old notebook are preserved in
[`archive/`](archive/README.md), with a hash manifest, and excluded from
distributions. Local reference repositories are not installation requirements.

The staged integration targets Hopper-or-newer GPUs with configurable cluster
resources and a CPU reference path. See [the roadmap](docs/superpowers/plans/2026-09-21-erot-flows-hpc-roadmap.md)
for implementation evidence and unexecuted hardware gates. See
[the changelog](CHANGELOG.md) for the local, unpublished 0.2 release. The original
EROT code retains MIT terms; adapted flow/QOTLib components use Apache-2.0, as
recorded in [NOTICE](NOTICE). SDPLab is independent and is not a dependency.
