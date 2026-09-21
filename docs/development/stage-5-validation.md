# Stage 5 experiment runtime verification

Historical integration record, captured before the subsequent repository cleanup.
See [the current repository map](repository-map.md) for the retained layout;
removed files and detailed task plans remain in EROT Git history.

The installed runtime now supports reproducible classical, quantum and flow
workers, atomic complete-state checkpoints, independent resource-isolated
processes and configurable Slurm arrays. Historical repositories and SDPLab
remain outside the runtime. The CPU environment is Python 3.12.13, JAX/jaxlib
0.11.2 and NumPy 2.5.3, with eight available affinity CPUs and no usable GPU.

Checkpoint tests exercise full complex quantum, classical and both flow states,
metadata compatibility, payload corruption and interruption before publication.
Worker tests compare uninterrupted and resumed complete states and preserve
failed flow state without advancing physical time. Launcher tests exercise one
and two real CPU workers, observed disjoint affinity, import-time isolation,
duplicate identity rejection, numerical failure exit status and explicit resume.
The array template has a local CPU execution test, plus manifest bounds and
relative-path checks. This is not actual scheduler validation.

The full CPU suite passed 238 tests with 9 skips (7 unavailable GPU tests and
2 opt-in artifact tests, run separately below). The launcher milestone passed
41 focused checks; timing audit tests passed 2.
Ruff checks and formatting passed for 96 Python files. Wheel and sdist built,
and both artifact inspections passed. Outside the checkout, a base-only wheel
installation ran erot-run for one classical chunk, then erot-launch --resume
completed the same identity using one assigned CPU. No optional reference
solver, optimizer or plotting package was required.

## CPU throughput smoke

The command in the experiment guide used the supplied classical-blocked and
quantum-entropy configurations, a fixed two-core budget, fresh outputs and
sequential worker-count trials. No other validation process was launched during
these trials. One repetition provides a smoke measurement, not variability or a
universal concurrency recommendation.

| Workers | CPU partition | End-to-end seconds | Completed runs/second |
|---|---|---:|---:|
| 1 | one worker with two cores | 8.127 | 0.2461 |
| 2 | two workers with one core each | 6.288 | 0.3181 |

Both experiments converged in both configurations. Timing coverage was complete:
5 classical chunks and 23 quantum chunks. With one worker, combined first-chunk
compile/execution was 1.130 seconds, subsequent execution 0.028 seconds and
checkpoint publication 0.801 seconds. With two workers these per-process sums
were 1.901, 0.024 and 0.788 seconds; sums are not parallel wall time. Worker setup
and interpreter/import/exit overhead are reported separately in the raw report.
These small cases are dominated by setup, compilation and checkpoint overhead.
The ignored raw record is benchmark-results/stage5/throughput/throughput.json;
all input configurations and the measurement script are versioned.

GPU worker throughput, GPU resource isolation on a real allocation, and actual
Slurm submission have not run. Configurable profiles and commands are ready for
those checks. No GPU speedup or capacity is inferred from these CPU results.

## Independent review and correction

One fresh reviewer found that a shorter final flow chunk could compile a new
static specialization while being labeled ordinary execution. The correction
tracks static count and state tree/shape/dtype/weak-type signatures. A seven-step
PDHG run with chunk length two reproduces the old misclassification and now
records compile/execution for the first and final variants, ordinary execution
for the repeated middle variants. The installed final wheel passes this check.
The final CPU suite passed 238 tests with 9 expected skips; final wheel/sdist
inspection passed 2 checks. No material review findings remain.

The lazy public import change allows the parent launcher to avoid loading JAX;
public API compatibility is tested. Real GPU isolation, Slurm launch and cluster
filesystem durability remain target-environment checks. Existing numerical
algorithms retain their earlier reference tests; this stage additionally tests
their worker integration. There were no deferred minor review findings.
