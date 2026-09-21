# Stage 4 verification

Historical integration record, captured before the subsequent repository cleanup.
See [the current repository map](repository-map.md) for the retained layout;
removed files and detailed task plans remain in EROT Git history.

The blocked two-marginal Sinkhorn path stores geometry, potentials and bounded
tiles instead of a full point-cloud cost/coupling. It streams updates, both
marginal residuals, Shannon objective and transport products. Explicit plan
blocks remain available. DenseGeometry preserves its existing dense input;
no memory claim is made for that storage.

Tests compare rectangular, nondivisible, zero-support and nonunit-mass cases
with the dense solver, including epsilon 0.01 and separate tiny-epsilon
diagonal cases. Exact continuation, changed-support warm starts, checked
iteration controls, invalid inputs, signed vector/matrix applications,
precision promotion, JIT and batching pass. Independent NumPy tile costs
include large common coordinate offsets to test cancellation.

The entropic flow path also accepts implicit geometry: its inner solve,
objective and transport mass term stream tiles. Two accepted steps match dense
trajectories. Failed inner solves retain physical state/time, and a compiled
129-cell chunk contains no full cost/coupling shape. PDHG remains dense and
rejects geometry objects explicitly.

The CPU benchmark and compiled memory evidence are recorded in
[blocked transport](../performance/blocked-transport.md). At fixed 32-by-32 tiles,
doubling both point counts changed blocked compiler buffer accounting from
53,492 to 95,092 bytes, versus 2,479,028 to 9,879,132 bytes for dense. Objectives
and convergence tolerances match. Dense is faster on these small CPU problems.
Compiler accounting is not runtime GPU peak memory; runtime peaks and tile
tuning on Hopper-or-newer devices remain unexecuted. Tools and precision-specific
CPU/GPU agreement tests are prepared.

Fresh review of 0362272..bb71309 found no important numerical or memory defects.
It reported a minor NumPy-backed geometry failure in direct calls, although
jit calls worked. The executor treated this as material because these floating
arrays pass validation and should not change validity under compilation.
Both geometry cases failed regression tests before conversion in tile access,
then passed. The review's remaining boundaries stand: overflowing coordinate
arithmetic requires rescaling, PDHG is dense, reverse differentiation through
dynamic solver loops is unsupported, and GPU measurements require hardware.

Final CPU verification: **172 passed, 9 skipped** (seven GPU cases and two
opt-in artifact tests). Ruff passes all 77 files. Rebuilt wheel/sdist content
tests pass separately (two tests). The rebuilt base-only wheel passes
NumPy-backed dense and point-cloud geometry solves, transport applications and
entropic flows outside the checkout. Historical flow and QOTLib checkouts are
clean; no SDPLab changes were made.

Important commits: 0326cdb (geometry/reductions), a2ccf07 (solver/outputs),
e43449c (memory tools/evidence), bb71309 (flow integration), followed by the
review correction. No native backend or measured GPU speedup is inferred.
