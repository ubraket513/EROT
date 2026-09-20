# Benchmarks and profiling

Run the planned dense benchmark matrix on the target GPU:

```bash
python benchmarks/benchmark_solvers.py \
  --device cuda:0 \
  --dtype float64 \
  --output benchmark-results/cuda-float64.json
```

Pass `--profile-dir benchmark-results/trace` to capture a JAX profiler trace.
The schema-2 JSON report separates first-call time from seven synchronized warm
samples, retaining their median/minimum/maximum and numerical outcomes. Each
CLI case runs in a fresh process; memory fields refer to the participating
device and distinguish active, peak, reserved, and capacity values when exposed.
Unavailable memory measurements remain null. The module invocation
`python -m benchmarks.benchmark_solvers` is also supported.

To enforce the 10% steady-state regression budget on the same reference
machine, retain an accepted JSON report and run:

```bash
python benchmarks/benchmark_solvers.py \
  --device cuda:0 \
  --baseline benchmark-results/accepted.json \
  --output benchmark-results/candidate.json
```

Only consider a custom Pallas/CuTe/C++ kernel after the trace identifies an
unsupported operation consuming at least 30% of total runtime. Keep it only if
it improves full-solve warm time by at least 1.5x or peak memory by at least
25% on the reference workload.

Comparisons reject changed objectives, inputs, dtype, geometry, tolerance,
output policy, hardware/affinity, or runtime, as well as failed solves and old
records without comparison identity. Source revision and a digest of the
working numerical/benchmark source are recorded; source may differ between
implementations. First-call measurements include compilation and execution,
but not subprocess startup.

See [the measurement protocol](../docs/performance/baseline-protocol.md).
Historical characterization tools are separate from performance winners:

```bash
python benchmarks/audit_legacy_flow.py --source-path /path/to/numerical-gradient-flows --output benchmark-results/legacy-flow.json
```

This audit requires its own environment with the historical dependencies.
It exits nonzero for mismatches/errors/unavailable checks and retains a JSON
report. No legacy repository or optimizer dependency enters the base package.
