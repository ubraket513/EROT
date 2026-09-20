# Development and release checks

Use the setup commands in [CONTRIBUTING.md](../CONTRIBUTING.md).
Python 3.11 and 3.12 are the initial CI targets. See
[the release validation record](development/stage-8-validation.md) for versions actually
exercised locally; minimum dependency combinations are not inferred from a
successful run with newer versions.

## Tests and formatting

```bash
python -m pytest
python -m pytest tests/unit tests/reference -m reference
python -m pytest tests/integration
ruff check src tests benchmarks examples experiments
ruff format --check src tests benchmarks examples experiments
```

The ordinary suite includes GPU tests that explicitly skip without CUDA.
Distribution-content tests skip unless `EROT_DIST_DIR` identifies built
artifacts. With a configured GPU environment, run `python -m pytest tests/gpu`.
CPU distributed tests start fresh virtual devices and two actual processes.
Use the pinned Python3.12 runtime in `requirements/distributed-cpu.txt` for these
experimental tests. Real GPU tests remain explicitly skipped without hardware.

CI separates lint, CPU unit/reference checks with coverage, CLI/data workflows,
distribution checks, and a small benchmark smoke run. Coverage is a diagnostic,
not proof of numerical correctness.

## Distribution checks

The default build command makes a source distribution and then builds a wheel
from that source distribution. Build another wheel directly from the checkout:

```bash
python -m build --wheel --outdir dist/direct
python -m build
EROT_DIST_DIR=dist EROT_DIRECT_DIST_DIR=dist/direct python -m pytest tests/packaging
```

Use a fresh artifact output directory if it contains old release versions.
The checks expect one top-level wheel and one source distribution. Inspect the
direct wheel through the same content checks and the clean installation below. Local reference
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
five console entry points, dense/blocked transport, PDHG flow, entropy QOT,
and generated-input/solve/archive CLI behavior. It requires plotting/reference/optimizer extras to be absent,
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
library constants. Experimental distributed solves have CPU correctness evidence; actual GPU and
cluster gates remain open. No custom native kernel is selected;
see the [architecture](superpowers/specs/2026-09-21-erot-flows-hpc-design.md).

## Compatibility and release evidence

Install the base minimum in a separate Python3.11 environment:

```bash
python -m pip install -c requirements/base-minimum.txt '.[test,plot]' 'cvxpy==1.6.5' 'pytest==8.3.5'
JAX_PLATFORMS=cpu python -m pytest tests/unit tests/reference tests/integration \
  --ignore=tests/unit/test_distributed_checkpoint.py \
  --ignore-glob='tests/integration/test_distributed*'
```

The distributed exclusions correspond to an explicitly newer runtime, not
unexplained base failures. Current CPU checks and installed artifacts are tested
separately. CI has minimum/current CPU, pinned distributed integration, lint,
packaging and benchmark jobs. No remote CI run or publication is implied by
local verification. Release candidates stay local until a separate publish
instruction; version metadata is in `pyproject.toml` and history in `CHANGELOG.md`.

Stage reports: [0](development/stage-0-validation.md),
[1](development/stage-1-validation.md), [2](development/stage-2-validation.md),
[3F](development/stage-3f-validation.md), [3Q](development/stage-3q-validation.md),
[4](development/stage-4-validation.md), [5](development/stage-5-validation.md),
[6](development/stage-6-validation.md), [7](development/stage-7-validation.md),
[8](development/stage-8-validation.md). The
[support matrix](support-matrix.md) is the public capability contract.
