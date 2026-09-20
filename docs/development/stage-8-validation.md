# Stage 8 release validation

The local release candidate is EROT0.2.0 on `chore/stage-0-cleanup` in the isolated
worktree. No remote branch, package index or release has been published or
merged. The original main checkout and reference repositories remain intact.
The [support matrix](../support-matrix.md) distinguishes CPU evidence from
prepared, unexecuted GPU/Slurm validation.

## Compatibility and distributions

The base minimum environment was exercised on Python3.11.15,
JAX/jaxlib0.4.30, NumPy1.26.4, SciPy1.12.0, ml-dtypes0.4.1,
opt-einsum3.3.0, CVXPY1.6.5 and pytest8.3.5. The command in
[development guidance](../development.md#compatibility-and-release-evidence)
passed **256 tests** in 158.13s. Experimental distributed tests are excluded
from this minimum job because they have a separately pinned newer runtime.
No failing base test was skipped to preserve the advertised minimum.

Both a direct wheel and a wheel rebuilt by the default sdist build passed
content inspection, alongside the sdist: **3 checks passed**. The direct-wheel
regression was demonstrated with an intentionally contaminated wheel; it
rejected an accidental checkout file outside `erot/` and distribution metadata.
Checks require integrated modules, all five console scripts, both licenses and
NOTICE, examples, HPC/config files, and exclusions for nested repositories,
archives and generated/tool state. Wheels remain pure Python.

Each wheel was installed into its own fresh base-only Python3.12 environment
under `/tmp/erot-stage8-installed/{direct,rebuilt}`. The copied
`tests/packaging/installed_smoke.py` ran with `-I` outside the checkout and verified:

- installed path and version metadata 0.2.0;
- all five console commands' help;
- classical dense product-plan reference and blocked transport convergence;
- an accepted PDHG step with preserved mass and changed density;
- entropy-QOT product-state reference;
- generated-input/solve/NPZ CLI workflow;
- absence of matplotlib, CVXPY, Flax and Optax in the base-only environments.

The fresh resolved CPU stack was Python3.12.13, JAX/jaxlib0.11.2,
NumPy2.5.3, SciPy1.18.1, ml-dtypes0.6.0 and opt-einsum3.4.0.
Raw local installation log: `/tmp/erot-stage8-installed.log`.

## Copied examples and installed workflows

All three scripts and supplied experiment configs were copied outside the
checkout and executed using the rebuilt wheel's interpreter/console commands:

- `quadratic_flow.py`: status0, three accepted steps, physical time1.2.
- `heat_flow.py --cells 16 --steps 2`: status0, accepted time0.04,
  mass error1.265e-8 and Neumann-mode L1 discrepancy0.0415963. This is a discrete
  smoke case; its nonzero continuum discrepancy is not concealed as PDE accuracy.
- `quantum_entropy.py`: status0 after14 accepted updates, marginal error5.869e-9,
  signed primal-dual difference-1.598e-9, minimum coupling eigenvalue0.099074.
- `erot-run classical-blocked.json`: deliberate stop after one chunk reports
  checkpointed, then resume completes with marginal error9.089e-9.
- `erot-launch` on the supplied quadratic-PDHG and quantum-entropy configs:
  two independent workers within four CPU slots complete, then explicit resume
  succeeds. The launcher partitions the allocation into two threads per worker.

Raw local results live under `/tmp/erot-stage8-installed/`, with console evidence
in `/tmp/erot-stage8-examples.log`. These are correctness/install smoke checks,
not isolated performance measurements.

## Preservation and release audit

All 11 entries in `archive/manifest.json` match their recorded byte sizes and
SHA256 hashes. Original EROT main remains at
`424c4ede55f67aa805c4f7914133a5a7bc1b9155`; numerical-gradient-flows remains at
`0c6f5b50aee9ab5378970e28d700ab8fd193f554`; QOTLib remains at
`bd534c61aeae082892b9b2421db153beb8e5c804`. Their tracked diffs are empty.
SDPLab is not touched, imported or packaged. Historical experimental classical
code is retained and explicitly not promoted by the new solver's evidence.

| Requirement | Authoritative evidence |
|---|---|
| Cleanup, recoverable data, installed layout, YACHT conventions | Archive hashes, Stage0 report, src discovery, artifact checks and CI configuration |
| Mathematical normalization and independent baselines | Stage1 audit reports, tests/reference, numerical convention documents |
| Composable transport and exact/warm state | Stage2 report, solver tests, state/checkpoint contracts |
| Gradient flow through shared EROT transport | Stage3F and Stage4 reports, joint-coupling references, blocked flow integration tests, copied examples |
| Selected QOTLib adaptation and attribution | Stage3Q report, complex/unequal-dimension tests, NOTICE and Apache license files |
| Memory-scalable classical representation | Stage4 compiler inspection and reference agreement; actual GPU live peaks explicitly unmeasured |
| Many experiments with reproducible restart | Stage5 tests, worker/launcher installed smoke, per-run config/state/timing records |
| Large distributed solve software | Stage6 virtual-device and two-process tests, local input/checkpoint/collective probes, prepared GPU/Slurm tools |
| Evidence-driven language choice | Stage7 decision,47 focused reporting tests, timing-scope regression; no native component selected |
| Independent release and migration | Both wheel paths, minimum runtime, copied examples, support matrix, migration0.2 and provenance |

Actual Hopper-or-newer solves, GPU isolated peak memory/block tuning, two-GPU
speed/capacity, communication share, Slurm submission, multi-node interconnect
and target-filesystem durability are **unexecuted**. User-confirmed lack of an
allocation changes this delivery to prepared tools and explicit open gates;
none is reported as a pass. Optional structured quantum research, distributed
PDHG, native components and end-to-end AD remain separate work packages.

## Final integrated verification

Fresh direct and sdist-derived wheels were built under `dist/stage8-final/`.
With both artifact directories configured, the complete current CPU suite passed
**289 tests, 9 skips** in 154.41s. All nine skips require unavailable CUDA hardware
(two specifically require two GPUs); distribution checks executed and passed.
Raw log: `/tmp/erot-stage8-final-suite.log`.

Ruff check and format-check pass across 114 Python files. `git diff --check`,
HPC shell syntax checks and release-document local link checks pass. Synthetic
native CLI comparisons return zero at1.5× and one at1.42857×, retaining both
input-file hashes; these are tool checks, not performance measurements.

Both final wheels replaced only EROT in the previously fresh base-only
environments; isolated installed smoke checks passed again. A second set of
fresh environments was attempted but hit the small `/tmp` filesystem's capacity;
the incomplete environment was removed before retrying with the verified
base-only environments. No dependency or test assertion was bypassed. Final
installation log: `/tmp/erot-stage8-final-installed.log`.

## Independent final review and decisions

The fresh final reviewer found no Critical or Important issue and independently
reran all three artifact checks. It verified both wheels' 51 package files against
current source and their installed copies, and inspected the minimum/current
suite and installed-smoke logs. All 51 current Python package files are present
byte-for-byte in both wheel flavors.

One minor follow-up is deferred: add an automated assertion that wheel `Version`
equals authoritative `pyproject.toml`, rather than only comparing installed
metadata with itself. Current direct/rebuilt wheel metadata and sdist metadata
were explicitly checked against `pyproject.toml` and all equal 0.2.0. This is a
future stale-artifact regression guard, not a current artifact defect.

Review rulings:

- Actual GPU/Slurm/interconnect/filesystem guarantees remain unmeasured, because
  the user authorized preparation without hardware. Cost if incorrectly assumed:
  unsupported platform/performance claims or unverified restart durability.
- Native performance/interface validation remains conditional on a selected
  component; this release selects none. Cost if later skipped: admitting an
  incorrect or slower backend.
- Earlier numerical stages rely on their independent references and the freshly
  executed full suite, not an additional duplicate run by the release reviewer.
  Cost if coverage misses a case: an undiscovered numerical defect; no broader
  numerical or hardware guarantee is inferred from packaging review.

The selected software/release roadmap is complete, with hardware gates explicitly
open as requested. The verified branch/worktree and local commits are preserved;
merging, pushing or publishing are separate operations and were not performed.
