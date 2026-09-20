# Source provenance

Recorded during Stage 0 on 2026-09-21. EROT remains an MIT-licensed project;
this stage imports no algorithm source from the reference repositories.

| Source | Inspected revision | License and role |
|---|---|---|
| EROT | `424c4ede55f67aa805c4f7914133a5a7bc1b9155` | MIT, existing package and artifacts |
| numerical-gradient-flows | `0c6f5b50aee9ab5378970e28d700ab8fd193f554` | Apache-2.0, confirmed by the user on 2026-09-21; future flow algorithm source |
| QOTLib | `bd534c61aeae082892b9b2421db153beb8e5c804` | Apache-2.0, confirmed by the user on 2026-09-21; future quantum algorithm source |
| YACHT | `48c86a09dee58f693f1223eb2b1a77764a62d9bd` | Engineering reference supplied by the user; no source copied |

The original EROT checkout was on `main` with no tracked modifications.
Its local `.serena/`, `numerical-gradient-flows/`, `QOTLib/`,
`SDP-Simulation/`, and planning documents were untracked.
Implementation uses the `chore/stage-0-cleanup` branch in an isolated worktree.
The original flow and QOTLib checkouts were clean and remain in place.
QOTLib's Git root is `QOTLib/QOTLib`; its recorded origin is
`git@github.com:Pavlo3P/QOTLib.git`. YACHT's supplied origin is
`git@github.com:ubraket513/YACHT.git`.

The Apache-2.0 declarations above are the user's explicit confirmation;
the inspected source trees did not contain tracked license files. The license
question raised during planning is resolved by that confirmation. When source
is adapted, record its original path, revision, authorship, destination,
modifications, and applicable license/NOTICE material alongside the adaptation.
Do not represent future Apache-2.0-derived files as solely EROT-authored MIT code.
This record does not modify the reference repositories or their histories.

The historical gradient-flow `pyproject.toml` contains template MIT metadata
and placeholder author details. For the planned adaptation, the user's explicit
Apache-2.0 confirmation is the recorded licensing instruction; the original
metadata is preserved unchanged and is not silently copied into EROT.

Historical EROT output paths, hashes, and destinations are recorded in
[the archive manifest](../../archive/manifest.json). Keep the original revision
and the migration commits so both original and reorganized content remain
recoverable.

## Initial CPU characterization

Python 3.12.13 on Linux x86-64, with eight available CPU affinity slots.
Dependencies used: JAX/jaxlib 0.11.2, NumPy 2.5.3, SciPy 1.18.1,
CVXPY 1.9.3, pytest 9.1.1, setuptools 84.0.0, and Ruff 0.16.8.
These are observed versions, not new minimum requirements.

The existing suite passed: 22 passed, 2 CUDA tests skipped. CLI help completed
successfully. The small CPU benchmark covers classical size 16 and quantum
subsystem size 2. Raw test output, environment versions, and before/after JSON
records are local artifacts under `benchmark-results/stage-0/`.
See the [Stage 0 validation record](../development/stage-0-validation.md) for
the final outcome. No GPU or cluster performance is inferred from these checks.

## Stage 3F flow integration

The new `src/erot/flows/` and `src/erot/optim/` components carry Apache-2.0
SPDX identifiers. The distribution records `MIT AND Apache-2.0`, retaining the
original EROT MIT terms and adding the full Apache license plus `NOTICE` to
both wheel and sdist. No original copyright or license file is removed.

The reference flow revision above records Dohyoung Ko as the last author of
`src/jko_lab/pdhg_jko.py`; its origin is
`git@github.com:ubraket513/numerical-gradient-flows.git`. The adaptation replaces
the historical class/function split with complete pure JAX state, dual-first
PDHG with the actual coupled operator norm, and explicit feasibility and
stationarity termination. Entropy prox uses a newly implemented log-coordinate
root solve instead of the historical overflow-prone Lambert-W expression.
Functionals use explicit nonnegative cell masses and volume-aware entropy.
The original checkout remains unchanged. QOTLib algorithm adaptation is still
a subsequent stage, with separate attribution required when added.
