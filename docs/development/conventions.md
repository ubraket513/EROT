# YACHT conventions adopted for EROT

License update (2026-09-21): the user confirmed Apache-2.0 for both QOTLib and numerical-gradient-flows. Earlier missing-license observations below describe the inspected trees, not an outstanding permission question. Preserve attribution and applicable notices when adapting source.

Status: the selected layout, packaging, style, testing and contributor conventions have been adopted. Native build systems and cluster containers remain conditional on selected capabilities/site validation. No YACHT code was copied into EROT.

Reference supplied by the user: `git@github.com:ubraket513/YACHT.git`.

Inspected revision: [`48c86a09dee58f693f1223eb2b1a77764a62d9bd`](https://github.com/ubraket513/YACHT/tree/48c86a09dee58f693f1223eb2b1a77764a62d9bd), commit dated 2026-04-26. A temporary local clone was inspected through Serena and direct configuration reads. No reference project dependencies were installed and its test suite was not run. This is a record of visible engineering patterns, not certification of the reference repository's release quality.

This record supplements the [EROT architecture](../architecture/design.md) and is implemented through the [delivery roadmap](roadmap.md), beginning with Stage 0 cleanup (historical record at commit `1d3ca9b`). QOTLib's algorithm candidates follow these engineering conventions; see the [separate adoption assessment](../architecture/qotlib-adoption.md).

## Evidence and adoption decisions

| Area | Observed in YACHT | EROT decision | Stage |
|---|---|---|---|
| Package layout | `src/yacht`, package discovery under `src` | Adopt `src/erot` while preserving imports | 0 |
| Packaging | Setuptools backend, `pyproject.toml`, console script | Retain setuptools; keep metadata authoritative in `pyproject.toml` | 0 |
| Native code | `src/cpp`, Makefile, custom build/install hooks, packaged executable | Keep an optional native boundary; choose build and wheel strategy only with the first accepted native component | 7 or early quantum branch |
| CLI | Top-level command dispatch with command-specific modules | Retain `erot` CLI and separate command parsing from numerical kernels | 0, 3F/3Q, 5 |
| Tests | Unit, bug-specific regression, integration, and complete CLI workflow tests | Adopt all four categories; add numerical reference, GPU, distributed, and restart tests | 0 onward |
| Fixtures | Checked-in small domain fixtures | Use small deterministic arrays and configuration fixtures; generate large numerical inputs | 0, 1 |
| CI | Install project, lint, execute CLI workflows, run pytest and report coverage | Adopt these practices and add isolated wheel/sdist installation checks | 0 |
| Style | Ruff configuration: 88 columns, four spaces, double quotes | Adopt formatting; target Python 3.11; enable additional correctness/import rules | 0 |
| Contribution practice | Docstrings, tests, documentation updates, focused changes | Adopt and expand numerical and performance review requirements | 0 |
| Environments | Conda environment, Dockerfile, conda recipe | Provide reproducible CPU and CUDA environment descriptions; validate cluster-compatible containers when cluster policy allows | 5 |
| User examples | Demo, use-case examples, documentation and paper material | Provide runnable OT/JKO examples and explicit citations to scientific formulations | 3F/3Q, 8 |

Evidence: [packaging](https://github.com/ubraket513/YACHT/blob/48c86a09dee58f693f1223eb2b1a77764a62d9bd/pyproject.toml), [native build hooks](https://github.com/ubraket513/YACHT/blob/48c86a09dee58f693f1223eb2b1a77764a62d9bd/setup.py), [Makefile](https://github.com/ubraket513/YACHT/blob/48c86a09dee58f693f1223eb2b1a77764a62d9bd/Makefile), [CI](https://github.com/ubraket513/YACHT/blob/48c86a09dee58f693f1223eb2b1a77764a62d9bd/.github/workflows/runTest.yml), [Ruff configuration](https://github.com/ubraket513/YACHT/blob/48c86a09dee58f693f1223eb2b1a77764a62d9bd/ruff.toml), and [contributor guidance](https://github.com/ubraket513/YACHT/blob/48c86a09dee58f693f1223eb2b1a77764a62d9bd/CONTRIBUTING_GUIDELINE.md).

## Adaptations needed for this numerical library

### Packaging and installation

The reference has both declarative metadata and custom install-time compilation. For EROT, retain a compiler-free base installation. A selected native backend must have an explicit optional installation path, a supported architecture/runtime matrix, and tests for both present and absent extensions. If shipping compiled wheels, verify platform tagging, dynamic dependencies, GPU architectures, and runtime diagnostics. An existing Fortran library can sit behind a C interface without becoming a mandatory dependency for every user.

Use `pyproject.toml` as the single version source initially and expose the installed version through distribution metadata. Do not introduce imports of solver code during package builds or merely to report the CLI version. Preserve the modern Python requirement instead of copying YACHT's historical Python bounds.

Release validation includes building from an sdist, inspecting wheel contents, installing outside the checkout, and exercising `erot --help` and a small numerical solve. These checks prevent a source-tree import from hiding a missing package file.

### Test isolation and scientific assertions

YACHT's workflow tests exercise the installed command and check domain results; its bug-specific tests preserve error behavior. EROT should follow that intent. Use `tmp_path`, independent fixtures, explicit subprocess argument lists, and bounded test workloads. Do not share mutable output directories between tests or depend on test execution order.

For OT, assert marginals, objective agreement, nonnegativity or PSD, and failure reporting. For JKO, additionally assert the stated time/cost convention, reference step accuracy, mass, and justified trajectory behavior. For GPU and distributed paths, compare scientifically meaningful values at declared tolerances rather than requiring bit-identical floating-point reductions. Exercise nonconvergence and restart failures as well as successful solves.

Coverage reporting is adopted. Numerical correctness still requires independent oracles and invariants, and performance claims require hardware measurements. Source inspection of a workflow file does not establish that CI currently passes.

### Coding conventions

Apply 88-column formatting, four-space indentation, double-quoted strings, snake_case functions/modules, and descriptive public types. Document public APIs and nontrivial numerical kernels with array shapes, real/complex dtype support, units, constraints, objective factors, and warm-start semantics. Keep derivations close to the mathematical documentation and cite them from implementations.

Use a Ruff Python 3.11 target. Start with `E4`, `E7`, `E9`, `F`, and `I` for syntax/correctness and imports, plus the formatter; exclude historical imported code until it has been migrated. Reformat in a dedicated change so numerical behavior reviews remain readable. Add type checking incrementally at the public/core boundary instead of claiming a complete repository-wide type guarantee before it is verified.

### Native integration

YACHT's C++ executable is an example of concentrating native work behind a Python interface. In EROT, launching a process and moving arrays through files for every inner GPU iteration would defeat the intended performance model. Use an in-process device operation where suitable, or a coarse-grained native solver invocation when it owns a complete distributed solve. Measure all transfers and communication at that boundary.

The presence of C++ in YACHT establishes a useful structural pattern; it does not establish that EROT needs a full rewrite. The architecture's profiling and capability criteria determine whether to introduce C++/CUDA or an existing Fortran library.

## Production-readiness evidence required in EROT

1. Public API and CLI contracts survive a clean installation.
2. Reference problems independently establish mathematical correctness.
3. Failed, nonfinite, and incompatible-input cases are explicit and tested.
4. Optional dependencies remain optional and their absence produces actionable errors.
5. Checkpoint/resume reproduces an uninterrupted run within declared tolerances.
6. CPU and Hopper-or-newer GPU capabilities are reported according to actual test evidence; untested successor hardware is identified explicitly.
7. Multi-GPU and multi-node claims include numerical and timing evidence. GPU/CPU counts and scheduler resources are configurable rather than fixed to one cluster.
8. Documentation provides installation, a small end-to-end example, numerical conventions, support limits, and contribution commands.
9. Source attribution and the histories of the original projects remain recoverable.

These are acceptance requirements for EROT. They are not assertions that every item already exists in YACHT or in the current EROT checkout.
