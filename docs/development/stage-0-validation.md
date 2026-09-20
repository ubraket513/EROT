# Stage 0 validation record

Date: 2026-09-21. Base revision:
`424c4ede55f67aa805c4f7914133a5a7bc1b9155`.
Implementation branch: `chore/stage-0-cleanup`.

## Results

| Check | Observed outcome |
|---|---|
| Original CPU suite | 22 passed; 2 CUDA tests skipped |
| After layout migration | 22 passed; same 2 CUDA skips |
| Version-metadata regression | Failed with the old hard-coded version; passed after metadata lookup |
| Full suite with distribution inspection | 25 passed; 2 CUDA skips |
| Python 3.11.15 compatibility suite | 23 passed; 2 CUDA skips and 2 opt-in artifact skips |
| Ruff correctness/import checks and formatting | Passed |
| Direct checkout wheel | Built and installed in a fresh base-only environment |
| Sdist-derived wheel | Built from the sdist and installed in a second fresh base-only environment |
| External installed smoke checks | Both passed: package location, version, console help, CPU solve, CLI generation/solve/archive |
| Archive preservation | All 11 files retain manifest SHA-256 values |
| Numerical source preservation | All numerical modules retain original Python syntax trees; only package version reporting changed |
| Coverage diagnostic | 80% of package statements in the full pytest run; not a numerical correctness certificate |

The CPU smoke benchmark retained these results exactly through the layout move:

| Problem | Size | Iterations | Residual | Converged |
|---|---|---|---|---|
| Classical Shannon | 16 | 76 | `9.231012948845569e-09` | Yes |
| Quantum quadratic | 2 | 28 | `7.4353839751850614e-09` | Yes |

No timing gain is claimed for restructuring. Compile and runtime samples remain
local reports, not a performance comparison.

## Reproduction

Install `.[test,plot,dev]` in an isolated environment and follow
[development checks](../development.md). The verification commands were:

```bash
JAX_PLATFORMS=cpu python -m pytest
ruff check src tests benchmarks
ruff format --check src tests benchmarks
python -m build --wheel --outdir dist/direct
python -m build
EROT_DIST_DIR=dist JAX_PLATFORMS=cpu python -m pytest --cov=erot --cov-report=term-missing
python benchmarks/benchmark_solvers.py --device cpu --classical-sizes 16 --quantum-sizes 2 --output benchmark-results/stage-0/after.json
```

Each wheel was installed into a distinct fresh environment containing only its
base dependencies. A copy of `tests/packaging/installed_smoke.py` ran from an
external temporary directory with `python -I` and `JAX_PLATFORMS=cpu`.
Its assertions verify that the imported package is under that environment's
prefix and that plotting/reference/optimizer extras are absent.

Observed Python/dependency versions are recorded in
[source provenance](../architecture/source-provenance.md).
Detailed local reports are under ignored `benchmark-results/stage-0/`.
The GitHub workflow itself has not been run remotely. GPU, multi-node, minimum
dependency, and native-backend claims remain outside this CPU verification.

## Next stage

An independent read-only review found no Critical or Important issues. It
verified the archived bytes/hashes, unchanged runtime modules and moved tests,
and artifact inspection. Both current wheels were independently inspected.
A minor follow-up is to automate content inspection for the direct wheel too;
the current automated content check selects the sdist-derived wheel, while
both wheels already receive isolated installation smoke tests. Stage 8 owns
that additional release check.

Resolved in Stage8: both direct and sdist-derived wheels now use the same
automated content checks via EROT_DIRECT_DIST_DIR and EROT_DIST_DIR, including
a deliberately contaminated direct-wheel regression.

Implementation decisions: use an isolated local worktree without changing
`main`; use existing behavior checks for mechanical moves and new regression
tests for changed version behavior. Remote CI execution, actual GPU/distributed
measurements, minimum dependency combinations, and deeper scientific audits
were not certified by this structural review.

Stage 1 adds independent mathematical contracts and historical flow/QOTLib
audits. Source import rights are recorded as Apache-2.0 for both projects,
per the user's confirmation. No QOTLib or gradient-flow algorithm was imported
during Stage 0, and neither original checkout nor SDPLab was changed.
