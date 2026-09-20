# EROT

EROT is a JAX-first research library for entropy-regularized classical and
quantum optimal transport. Its stable API currently provides:

- multi-marginal log-domain Sinkhorn for Shannon-regularized classical OT;
- two-marginal cyclic dual projection for quadratic classical OT;
- Dykstra cyclic projection for quadratic quantum OT with general Hermitian
  density matrices and positive-semidefinite couplings.

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
input conversion. Use `dtype="float32"` for faster execution on GPUs with
limited double-precision throughput.

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

erot plot result.npz --output coupling.png
```

Use `erot solve quantum` for general density matrices; quadratic regularization
and cyclic projection are selected automatically. Solver results use `.npz`
and contain the coupling, residual, iteration count, convergence flag, elapsed
time, and JSON metadata. Plotting is optional and requires `EROT[plot]`.

## Numerical and performance notes

The full iteration loops execute through `jax.lax.while_loop` inside `jax.jit`.
The first solve for a new shape includes compilation; later solves reuse the
compiled executable. Benchmark compilation and warm runtime separately and
call `block_until_ready()` before stopping a GPU timer.

Dense multi-marginal costs require memory exponential in the number of
marginals. Dense quantum couplings require `O((nm)^2)` storage and an
eigendecomposition per iteration. Sparse, matrix-free, and distributed methods
are outside the 0.1 release.

See [the migration guide](docs/migration-0.1.md) when upgrading from the legacy
`eot` package and PyTorch `.pt` files.

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
for planned flow, quantum, and distributed capabilities; these are not claims
about features already present in version 0.1.0.
