# Changelog

## 0.2.0 — integrated research release (unpublished)

- Adopt a src layout, isolated wheel/sdist checks, reference/integration tests,
  Ruff conventions and source/artifact provenance inspired by YACHT.
- Expose pure JAX solver state, cost-unit Sinkhorn potentials, explicit warm
  starts and complete resume state while retaining the dense `erot.solve` API.
- Integrate entropy, quadratic and potential gradient flows with PDHG and
  finite-epsilon entropic JKO, corrected normalization/prox rules, diagnostics,
  physical-time accounting and separate snapshots/checkpoints.
- Add complex unequal-dimension quantum operators and dense von Neumann entropy
  transport, with independent primal/dual references and Apache-2.0 attribution
  for adapted numerical-gradient-flows and QOTLib components.
- Add blocked point-cloud transport and entropic flows without a dense cost or
  plan, including streamed objectives and plan application.
- Add reproducible experiment workers, resource-aware independent launchers,
  atomic validated checkpoints, Slurm array templates and throughput tools.
- Add experimental row-sharded Sinkhorn, coordinated shard checkpoints,
  configurable distributed launch and profiler/scaling tools. CPU virtual-device
  and two-process tests pass; actual GPU/Slurm/interconnect validation is pending.
- Keep JAX as the selected implementation. Prepare evidence gates for optional
  native kernels; no C++/CUDA/Fortran component is selected without measurements.

See [migration](docs/migration-0.2.md), [support](docs/support-matrix.md), and
[stage validation records](docs/development.md) for scientific changes and limits.
This is a local release candidate; no package or remote release was published.

## 0.1.0

Existing JAX classical Shannon/quadratic and quantum quadratic transport API,
CLI and NumPy result archives. See the [legacy migration guide](docs/migration-0.1.md).
