# Development and release checks

Use the setup commands in [CONTRIBUTING.md](../CONTRIBUTING.md).
Python 3.11 and 3.12 are the initial CI targets. See
[the validation record](development/stage-0-validation.md) for versions actually
exercised locally; minimum dependency combinations are not inferred from a
successful run with newer versions.

## Tests and formatting

```bash
python -m pytest
python -m pytest tests/unit -m reference
python -m pytest tests/integration
ruff check src tests benchmarks
ruff format --check src tests benchmarks
```

The ordinary suite includes GPU tests that explicitly skip without CUDA.
Distribution-content tests skip unless `EROT_DIST_DIR` identifies built
artifacts. With a configured GPU environment, run `python -m pytest tests/gpu`.
The `distributed` marker is reserved for actual distributed-runtime tests;
none are implemented by this structural migration.

CI separates lint, CPU unit/reference checks with coverage, CLI/data workflows,
distribution checks, and a small benchmark smoke run. Coverage is a diagnostic,
not proof of numerical correctness.

## Distribution checks

The default build command makes a source distribution and then builds a wheel
from that source distribution. Build another wheel directly from the checkout:

```bash
python -m build --wheel --outdir dist/direct
python -m build
EROT_DIST_DIR=dist python -m pytest tests/packaging
```

Use a fresh artifact output directory if it contains old release versions.
The checks expect one top-level wheel and one source distribution. Inspect the
direct wheel as well through the clean installation below. Local reference
repositories, archive outputs, checkpoints, environments, and tool state must
not enter release artifacts. The base wheel remains pure Python and needs
neither a native compiler nor a CUDA toolkit.

Run the following from the repository root to exercise both build paths outside
the checkout (POSIX shell):

```bash
check_root="$(mktemp -d)"
cp tests/packaging/installed_smoke.py "$check_root/installed_smoke.py"
for flavor in direct rebuilt; do
  python -m venv "$check_root/$flavor"
  if [ "$flavor" = direct ]; then
    "$check_root/$flavor/bin/python" -m pip install dist/direct/*.whl
  else
    "$check_root/$flavor/bin/python" -m pip install dist/*.whl
  fi
  (cd "$check_root" && JAX_PLATFORMS=cpu "$check_root/$flavor/bin/python" -I installed_smoke.py)
done
```

The smoke check verifies installed package location, version metadata, the
console entry point, a CPU reference solve, and generated-input/solve/archive
CLI behavior. It requires plotting/reference/optimizer extras to be absent,
so accidental mandatory dependencies are visible. Keep the temporary directory
for inspection, or remove that specific directory when finished.

The version in `pyproject.toml` is authoritative. `erot.__version__` reads
installed distribution metadata; changing the version requires reinstalling
the editable package or rebuilding the wheel.

## Hardware policy

The optimization target is Hopper-or-newer NVIDIA GPUs, while the base
installation and reference tests remain usable on CPU. Publish actual tested
Python/JAX/jaxlib/CUDA/driver/device combinations and distinguish validated,
experimental, and untested paths. A newer GPU is not automatically a validated
custom-kernel target.

GPU counts, CPU affinity/thread budgets, memory, node counts, interconnect,
accounts, partitions, and QoS will be allocation/profile settings rather than
library constants. Stage 0 does not implement distributed solves or custom
kernels. Those stages require measured numerical and hardware evidence;
see the [architecture](superpowers/specs/2026-09-21-erot-flows-hpc-design.md).
