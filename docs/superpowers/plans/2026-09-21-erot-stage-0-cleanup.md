# EROT Stage 0: repository cleanup and production structure

License update (2026-09-21): the user confirmed Apache-2.0 for both QOTLib and numerical-gradient-flows. Earlier missing-license observations below describe the inspected trees, not an outstanding permission question. Preserve attribution and applicable notices when adapting source.

Status: Stage 0 completed and independently reviewed on 2026-09-21. See [validation evidence](../../development/stage-0-validation.md).

> **For agentic workers:** Use `superpowers:executing-plans` when implementing inline. Complete one reviewable task at a time; checkboxes below describe future work.

**Goal:** establish a clean, independently installable EROT repository before importing flow or QOTLib algorithms. Preserve existing numerical behavior and source provenance while adopting the [YACHT conventions](../specs/2026-09-21-yacht-conventions.md).

**Scope:** inventory, recoverable artifact organization, `src/` packaging, test placement, style/tooling, and contributor documentation. Solver extraction, algorithm corrections, QOTLib copying, HPC optimizations, and SDPLab changes belong to later work packages. The [roadmap](2026-09-21-erot-flows-hpc-roadmap.md) and [architecture](../specs/2026-09-21-erot-flows-hpc-design.md) define those boundaries.

**Environment:** CPU Python >=3.11 with JAX/NumPy, existing test/plot dependencies, and isolated packaging tools. A GPU allocation is not required to complete this structural stage. Record exact versions used; do not change dependency floors casually while moving files.

## Proposed path mapping

| Current path | Stage 0 destination or treatment |
|---|---|
| `erot/` | `src/erot/`, preserving module names, numerical bodies, and public imports |
| `tests/test_api.py`, `test_classical.py`, `test_quantum.py`, `test_generate.py` | Same basenames under `tests/unit/`; retain existing reference markers within mixed modules initially |
| `tests/test_cli_and_io.py` | `tests/integration/test_cli_and_io.py`; use temporary output directories |
| `tests/test_gpu.py` | `tests/gpu/test_gpu.py`; preserve explicit hardware skips |
| `demo/{C,X,Y}.pt`, `demo/readme.md` | `archive/legacy-erot/demo/` after checking references and recording hashes |
| `tests/demo_eqiuv_dist/`, `tests/demo_eqiuv_distribution/`, `tests/demo_result/` | `archive/legacy-erot/tests/` with original directory names; these are historical outputs, not assumed test fixtures |
| Tracked `test.ipynb` | `archive/legacy-erot/test.ipynb`; preserve bytes, document historical dependencies |
| `benchmarks/` and existing user docs | Remain top-level, with paths corrected where required |
| `.serena/` | Local tool state excluded from version control and distributions |
| `numerical-gradient-flows/`, `QOTLib/` | Retain local checkouts and nested `.git` histories, explicitly exclude them from EROT packaging/version control; import selected components only in later stages |
| `SDP-Simulation/` | Remains untouched and outside EROT's package/release inputs |

Archive paths retain recoverable historical files during this migration and are excluded from wheel/sdist artifacts. The public repository can later retire archived generated outputs in a separate reviewed change once provenance and reproduction are adequate. Do not delete original files merely because a name looks obsolete.

## Task 1: inventory and capture existing behavior

**Files:** create `docs/architecture/source-provenance.md` and `docs/development/repository-map.md`; write local reports under ignored `benchmark-results/stage-0/`.

- [x] Record EROT's branch/revision and dirty state. Inventory tracked and untracked paths separately using `git ls-files` and `git status`; record nested flow and QOTLib revisions/status without staging them.
- [x] Confirm source origin, attribution, and applicable license. Currently neither the flow checkout nor QOTLib has a discovered tracked license file. Keep their source out of the public package until reuse is established; this does not block reorganizing EROT's own files.
- [x] Record the YACHT reference revision `48c86a09dee58f693f1223eb2b1a77764a62d9bd`. Record QOTLib's actual Git root as `QOTLib/QOTLib`, not its outer directory.
- [x] Create a manifest for every historical file to move: original path, destination, size, SHA-256, whether tracked, and source revision. Search README, docs, code, tests, and workflows for dependencies on those paths. A required fixture stays active or gets an explicit corrected reference.
- [x] Establish an isolated CPU environment, preserving any pre-existing environment. Install the current package and its declared test dependencies, then record the existing suite, CLI help, and small CPU benchmark before moving source files.

Representative commands, executed individually from the root with the selected interpreter:

```bash
git status --short
git rev-parse HEAD
git ls-files
git -C numerical-gradient-flows status --short
git -C numerical-gradient-flows rev-parse HEAD
git -C QOTLib/QOTLib status --short
git -C QOTLib/QOTLib rev-parse HEAD
python -m venv .venv-cleanup
.venv-cleanup/bin/python -m pip install -e '.[test,plot]'
.venv-cleanup/bin/python -m pip freeze
.venv-cleanup/bin/python -m pytest
.venv-cleanup/bin/erot --help
.venv-cleanup/bin/python benchmarks/benchmark_solvers.py --device cpu --classical-sizes 16 --quantum-sizes 2 --output benchmark-results/stage-0/before.json
```

These are compatibility snapshots, not the full Stage 1 numerical or performance campaign. Record failures and skips truthfully. If a needed dependency cannot be installed, distinguish that environment limitation from a solver regression. Preserve pre-existing changes; do not use `git clean`, destructive resets, or blanket staging.

**Exit:** all affected paths have a disposition, historical source is recoverable, and the existing local behavior has a recorded outcome.

## Task 2: separate historical outputs and local checkouts

**Files:** add `archive/README.md` and a machine-readable artifact manifest; move only the explicitly mapped tracked artifacts. Update `.gitignore` and the repository map.

- [x] Move tracked historical artifacts with rename-preserving operations. Confirm hashes after each move and update any verified references. Do not rewrite binary contents or strip notebook cells as part of this task.
- [x] Keep current local clones in place and add explicit root ignore entries for `/numerical-gradient-flows/`, `/QOTLib/`, `/SDP-Simulation/`, and `/.serena/`. Explain in the repository map that these are local source references, not vendored dependencies or required checkout contents.
- [x] Ignore local virtual environments, generated reports, result/checkpoint directories, and build output. Keep deliberate tiny fixtures trackable; avoid global exclusions that hide legitimate scientific input files.
- [x] Document how a fresh contributor works with EROT alone. Legacy audit instructions take an explicit external source path; do not require every user to clone nested projects.
- [x] Verify that the parent index contains no gitlink or nested `.git` content and that nested repository status is unchanged. Avoid resolving this by moving or deleting SDPLab.

**Exit:** active tests/demo paths contain only intentional inputs/workflows, generated files have recoverable locations, and EROT's release inputs are separated from local reference projects.

## Task 3: move the package and tests without changing algorithms

**Files:** move `erot/` to `src/erot/`; update `pyproject.toml`, add `MANIFEST.in` if needed to enforce sdist exclusions, move test modules according to the mapping, and correct existing documentation/CI paths.

- [x] Change setuptools discovery to `where = ["src"]` and retain the `erot*` package selection, console script, project name, and supported Python requirement. Verify subpackages such as `erot.experimental` remain included.
- [x] Preserve `erot.solve`, `SolverConfig`, result fields, public module imports, CLI flags, and serialized-data behavior. Do not split `quantum.py` into a package or change solver signatures while moving the tree.
- [x] Move tests without renaming test functions or weakening assertions. Keep mixed reference/unit modules intact initially; Stage 1 adds dedicated independent references under `tests/reference/`.
- [x] Ensure tests and benchmarks import the installed distribution. Do not add a `sys.path` escape or `PYTHONPATH=src` to hide broken discovery. Repository-only measurement modules remain outside the runtime distribution.
- [x] Preserve version `0.1.0` during the move. Version-source consolidation can follow in the tooling task, with an installed-metadata smoke check; no release/version bump is implied.
- [x] Reinstall the editable package and rerun the same compatibility checks. Compare numerical outcomes, test counts/statuses, CLI behavior, and benchmark case outputs against Task 1 rather than asserting identical wall times.

**Exit:** imports and behavior match the snapshot, expected test discovery is preserved, and the structure follows `src/` without numerical changes.

## Task 4: make installed artifacts and engineering checks reliable

**Files:** update `.github/workflows/runTest.yml`, `pyproject.toml`, and optional manifest configuration; add `ruff.toml`, `CONTRIBUTING.md`, `docs/development.md`, and focused packaging/workflow checks where needed.

- [x] Add explicit development/build tooling dependencies and keep optional plotting/reference/native dependencies optional. Follow the existing runtime requirement scope; do not inherit QOTLib's research dependency set.
- [x] Adopt Ruff's 88-column, four-space, double-quote formatter, Python 3.11 target, and initial `E4`, `E7`, `E9`, `F`, `I` checks. Apply mechanical formatting separately from renames and scientific changes. Review import ordering for initialization side effects.
- [x] Make `pyproject.toml` authoritative for version reporting and use installed metadata without importing numerical solver code during builds. Preserve public `erot.__version__` behavior with a focused installed check.
- [x] Build wheel and sdist, inspect both file lists, and explicitly reject legacy checkouts, archive binaries, generated reports, environments, and machine-specific configuration in distribution artifacts. Include required license, package modules, and intentional package data.
- [x] Build a wheel from the extracted sdist as well as from the working source. Install each resulting wheel into a separate clean environment. Run from a temporary directory outside the checkout with no inherited source path and assert `erot.__file__` is inside the selected environment.
- [x] Run import, version, CLI help, a small CPU solve, and an installed CLI workflow from those environments. Copy only required tests/fixtures into the external test directory; assert output correctness as well as command exit status. Verify base import/help without plotting, native tooling, or CUDA.
- [x] Keep CPU unit/reference, installed workflow, formatting/lint, and package checks as distinct CI responsibilities. GPU/distributed tests are opt-in and must not count CPU simulations as GPU validation. Keep the currently tested Python versions initially and document untested combinations.
- [x] Document local commands, source/layout boundaries, test markers, mathematical docstring expectations, and the Hopper-or-newer support policy. No cluster credentials or site-specific account names belong in reusable configuration.

**Exit:** both artifact construction paths install correctly outside the source tree, existing numerical checks retain their outcomes, style checks pass, and package contents are intentional. Any existing scientific failure remains identified for Stage 1/solver work, not relabeled as a packaging success.

## Task 5: review and hand off the clean foundation

- [x] Review rename, artifact, packaging, and formatting diffs separately. Confirm no numerical expressions, stopping rules, defaults, or external repositories changed.
- [x] Recheck the artifact manifest and nested repository status; ensure original tracked content is recoverable.
- [x] Verify links, contributor commands, clean artifact installation, test discovery, and the absence of accidentally staged local clones.
- [x] Record the before/after CPU validation outcomes and remaining scientific/license/environment questions. Do not claim benchmark speedups from a layout change.
- [x] Update roadmap status with actual evidence and proceed to the [Stage 1 numerical/hardware baseline plan](2026-09-21-erot-stage-1-baselines.md).

The first implementation milestone is this clean, installable foundation. No algorithm import, native rewrite, remote push, repository deletion, or publication is required to complete it.
