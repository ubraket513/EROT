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

Consider a custom Pallas/CuTe/C++ kernel only after profiling identifies a
bottleneck (initial screening: at least 30% of total runtime), or a documented
required capability justifies one. Keep it only if
it improves full-solve warm time by at least 1.5x or peak memory by at least
25% on the reference workload.

Comparisons reject changed objectives, inputs, dtype, geometry, tolerance,
output policy, hardware/affinity, or runtime, as well as failed solves and old
records without comparison identity. Source revision and a digest of the
working numerical/benchmark source are recorded; source may differ between
implementations. First-call measurements include compilation and execution,
but not subprocess startup.

See [the measurement protocol](../docs/performance/baseline-protocol.md).
Historical one-time source audits are recorded in the numerical audit documents;
their removed scripts are available from Git commit `1d3ca9b`. Active benchmark
tools require only the integrated EROT package and their documented dependencies.

Hardware inventory and isolated quantum projection:

```bash
python -m benchmarks.hardware_inventory --backend cpu --output benchmark-results/hardware.json
python -m benchmarks.benchmark_quantum_projection --device cpu --dimension 16 --dtype complex128 --output benchmark-results/projection.json
```

Use `gpu` explicitly on an allocated CUDA machine; unavailable GPU execution
fails instead of being counted as CPU validation. See
[quantum feasibility](../docs/performance/quantum-feasibility.md) for the workload
ladder, memory accounting, native backend boundary and unexecuted hardware gates.

Integrated release tools and evidence:

- [Blocked classical solves](../docs/performance/blocked-transport.md): compiler
  allocation inspection and synchronized dense/blocked comparisons.
- [Independent experiment throughput](../docs/numerics/experiments.md):
  `experiment_throughput.py`, including startup, compilation and checkpoint costs.
- [Distributed solves](../docs/performance/distributed-transport.md):
  `distributed_sinkhorn.py`, rank-local reports and optional profiler traces.
- [Native admission](../docs/performance/native-decision.md): `native_gate.py`,
  conditional Amdahl screening and matched full-solve measurement gates.

Actual GPU/Slurm results remain unmeasured. Each protocol identifies which CPU
measurements exist and which hardware gates still need an allocation.
