# Contributing to EROT

Install a development environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,plot,dev]'
python -m pytest
ruff check src tests benchmarks examples experiments
ruff format --check src tests benchmarks examples experiments
```

The package uses `src/erot/`; install it before running tests. Do not add
`PYTHONPATH=src` or mutate test import paths to compensate for missing package
discovery. The base package requires JAX and NumPy. Plotting, numerical
references, development tools, and any future native backend remain optional.

Keep file moves, formatting, mathematical corrections, and performance changes
in separate commits. Preserve public imports and CLI behavior unless a change
is explicitly documented. Use focused local commits and include the verification
commands and outcomes in the change description.

Use Python 3.11-compatible syntax, typed public interfaces, snake_case names,
four-space indentation, double quotes, and Ruff's 88-column formatter.
Numerical docstrings should state shapes, dtypes, mass/trace conventions,
objective factors, stopping criteria, and supported JAX transformations.
Document the distinction between warm starts and exact resume state.

Tests should assert numerical objectives/invariants and explicit failure
behavior, use independent small references where needed, and write generated
data under `tmp_path`. Avoid shared mutable output directories and shell-built
subprocess commands. GPU tests must skip explicitly when CUDA is unavailable;
CPU tests do not establish GPU performance.

Before adapting external source, record its revision, authorship, license,
original path, and destination. The user confirmed Apache-2.0 for QOTLib and
numerical-gradient-flows; preserve their applicable license and notices when
importing source. See [source provenance](docs/architecture/source-provenance.md).

See [development and release checks](docs/development.md),
[repository layout](docs/development/repository-map.md), and
[the implementation roadmap](docs/superpowers/plans/2026-09-21-erot-flows-hpc-roadmap.md).
