# Repository map

EROT is one independently installable library at the repository root. No nested
source checkout, archived demo dataset or separate SDP project is required.

| Path | Purpose |
|---|---|
| `src/erot/` | Supported API/CLI, solvers, geometry, flows, operators and runtime |
| `tests/unit/`, `tests/reference/` | Behavioral checks and independent numerical oracles |
| `tests/integration/` | CLI, worker, restart and runtime checks |
| `tests/distributed/`, `tests/gpu/` | Distributed subprocess probes and hardware-dependent checks |
| `tests/packaging/` | Artifact contents and installed-package smoke checks |
| `benchmarks/` | Active correctness-aware performance and hardware tools |
| `examples/` | Runnable gradient-flow and quantum examples |
| `experiments/configs/` | Reproducible workloads for installed `erot-run`/`erot-launch` commands |
| `hpc/` | Configurable launch templates and allocation profiles |
| `requirements/` | Base-minimum and experimental distributed CPU environments |
| `docs/` | Numerical contracts, support, architecture, migration and validation evidence |

Generated outputs, logs, local AI-agent instructions/plans, environments and
tool caches are ignored. Maintained documentation and CI definitions stay tracked. Completed detailed
task plans, legacy audit tools, archived demos and unused experimental solvers
are available in EROT Git history before cleanup. Original QOTLib,
numerical-gradient-flows and SDPLab local checkouts were deleted as requested;
source revisions and required attribution remain in
[provenance](../architecture/source-provenance.md) and [NOTICE](../../NOTICE).

See [contributing](../../CONTRIBUTING.md), [support](../support-matrix.md),
and [future work](roadmap.md).
