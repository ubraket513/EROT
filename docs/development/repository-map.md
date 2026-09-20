# Repository map

EROT is the independently installable numerical library. A fresh contributor
needs only this repository and its declared development dependencies.

| Path | Purpose |
|---|---|
| `src/erot/` | Public API, numerical implementation, CLI, experimental modules |
| `tests/unit/` | Existing API/algebra tests, including marked independent references |
| `tests/integration/` | CLI and data workflow checks with temporary outputs |
| `tests/gpu/` | Explicitly GPU-dependent checks |
| `tests/packaging/` | Distribution-content and installed-package guarantees |
| `benchmarks/` | Repository tools for reproducible measurements |
| `docs/` | User migration, architecture, contributor, and staged implementation records |
| `archive/legacy-erot/` | Byte-preserved historical demonstrations and generated outputs |
| `archive/manifest.json` | Original paths, destinations, hashes, sizes, and source revision |

Independent numerical references live under `tests/reference/`; fresh-process
distributed probes under `tests/distributed/`. Active solver, geometry, flow,
operator and runtime modules live under `src/erot/`. `experiments/configs/`
contains reproducible workloads; `hpc/` holds configurable launch templates,
and `requirements/` distinguishes base-minimum and experimental distributed
CPU runtimes. See the [support matrix](../support-matrix.md).

Optional local checkouts `numerical-gradient-flows/`, `QOTLib/`, and
`SDP-Simulation/` are ignored by the parent repository and excluded from
distributions. They are not Git submodules or runtime dependencies.
Their original locations and histories are preserved. Future historical audit
tools must accept an explicit source path; ordinary EROT tests must not depend
on these checkouts. SDPLab stays independent.

Local environments, Serena/worktree state, benchmark reports, and generated
checkpoints/results are ignored. Intentional small fixtures remain trackable.
The archive contains historical inputs and outputs, not active test fixtures;
it is excluded from both wheel and source distributions.

See [source provenance](../architecture/source-provenance.md),
[contributor instructions](../../CONTRIBUTING.md), and
[the cleanup plan](../superpowers/plans/2026-09-21-erot-stage-0-cleanup.md).
