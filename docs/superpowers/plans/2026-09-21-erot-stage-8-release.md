# Stage 8 integrated release implementation plan

> Execute inline with superpowers:executing-plans under the user's full-roadmap
> authorization. One fresh final review; no push, publication or merge.

**Goal:** deliver a locally committed, independently installable integrated 0.2
release with verifiable packaging, compatibility, migration and support claims.
**Architecture:** keep src/erot and the current public wrappers; extend artifact
inspection to both build paths, separate base and experimental distributed
runtime testing, and publish the evidence through concise user/developer guides.
**Tech stack:** setuptools, pytest, Ruff, JAX/NumPy, existing examples and CLIs.
**Spec:** ../specs/2026-09-21-erot-flows-hpc-design.md and roadmap Stage8.

## Global constraints

Python >=3.11; current declared base JAX >=0.4.30 and NumPy >=1.26 must be tested
or corrected with evidence. Existing MIT terms and adapted Apache-2.0 attribution
remain intact. User-approved lack of hardware means GPU validation tools and
explicit unexecuted gates, never fabricated GPU results. Preserve original
reference repositories and SDPLab independence. No remote mutation.

## Review focus

A direct wheel may differ from a wheel rebuilt from sdist; inspect and install both.
A dependency floor may permit a version missing APIs used by a newer feature.
Examples can accidentally import the checkout instead of the installed wheel.
Migration must expose scientific changes, not just rename old functions.
Hardware readiness must not be represented as tested GPU or Slurm support.

## Task 1: artifact and runtime compatibility contract

Files: tests/packaging/test_distributions.py, tests/packaging/installed_smoke.py,
pyproject.toml, MANIFEST.in, requirements/base-minimum.txt,
.github/workflows/runTest.yml.

- [x] Add parametrized direct/rebuilt wheel inspection. Read optional
  EROT_DIRECT_DIST_DIR alongside required EROT_DIST_DIR; when supplied require
  exactly one wheel and apply the same content/license/entry-point assertions.
  Demonstrate RED using an invalid direct wheel, then GREEN using fresh builds.
- [x] Require integrated flow/geometry/quantum/distributed package files, all
  five console scripts, version/license metadata, and shipped examples/HPC
  configs/docs in artifacts. Retain exclusions for archived/nested/generated data.
- [x] Set authoritative version0.2.0 and include changelog in sdist. Keep alpha
  status for the research release; no native backend is selected.
- [x] Run base minimum Python3.11/JAX0.4.30/NumPy1.26.4 in an isolated environment;
  pin compatible SciPy/ml-dtypes test dependencies. Run numerical/core/worker
  tests excluding explicitly newer distributed runtime. Fix real incompatibility
  or change supported bounds with evidence, never silently skip base failures.
- [x] CI separates minimum CPU, current CPU and pinned experimental distributed
  checks, and inspects both wheel flavors. No self-hosted GPU job assumed.
- [x] Extend base-only installed smoke to integrated flow/entropy-QOT and console
  availability; execute outside checkout with -I and no optional optimizer/plot/
  reference packages. Commit verified packaging/compatibility changes.

## Task 2: migration and public support documentation

Files: README.md, CHANGELOG.md, docs/migration-0.2.md, docs/support-matrix.md,
docs/development.md, docs/architecture/source-provenance.md.

- [ ] Document actual CPU/version evidence, GPU target versus validation status,
  distributed experimental runtime, objectives, precision, transformations,
  capacity, unsupported derivatives and quantum spectral limits.
- [ ] Map legacy jko_lab PDHG/Sinkhorn/functionals/history into erot.flows and
  full state/checkpoints. Record mass-volume/cost-factor/prox/termination changes.
- [ ] Map adopted QOTLib operators/spectral entropy/dual methods; state unequal
  dimensions, positive-definite marginal restriction, conjugate correction,
  non-adopted chordal/Lanczos/backends and provenance/license locations.
- [ ] Update README0.2 features, base/GPU installation, runnable examples and
  experiment/distributed documentation links; remove stale0.1 scalability claims.
- [ ] Document reproducible benchmark/profile/native-admission tools, future GPU
  commands, configurable Slurm launch and actual unexecuted gates.
- [ ] Account for retained experimental code and preserved historical repositories;
  do not delete capabilities merely to make packaging simpler. Commit docs.

## Task 3: release evidence, review and completion audit

Files: docs/development/stage-8-validation.md, roadmap, relevant fixes.

- [ ] Build fresh direct wheel and sdist-derived wheel; inspect both; install
  each in separate clean base-only environments and run installed_smoke.py.
- [ ] Copy all documented example scripts and supplied experiment configurations
  outside checkout; run heat_flow, quadratic_flow, quantum_entropy, installed
  worker and launcher resume smoke. Record scientific outcomes, not exit alone.
- [ ] Run full current CPU/reference/integration suite, minimum suite, Ruff and
  all artifact checks. Record explicit GPU/artifact skips, never as passes.
- [ ] Obtain one fresh final review for Stage8, reproduce/fix material findings,
  then rerun affected checks/full suite as warranted and commit.
- [ ] Audit every roadmap/software deliverable against current files and evidence;
  retain hardware gates explicitly open per the user's preparation instruction.
  Confirm original/nested Git status and SDPLab untouched. Preserve branch and
  commits for user review; do not merge/publish. Mark goal achieved only when
  every selected software/release requirement is verified.

## Self-review

Tasks cover Stage8, the deferred direct-wheel inspection, declared dependency
floor, migration, provenance, scientific/support limits and final evidence.
No architectural redesign or optional research backend enters release cleanup.
