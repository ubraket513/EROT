# Baseline protocol

Stage 1 measures validated numerical behavior before performance decisions.
The first available environment is CPU-only; the user confirmed that no
Hopper-or-newer allocation is accessible yet. GPU validation tools and cases
can be prepared, but GPU timing, memory, communication, and native-backend
claims remain unmeasured.

## Dense solver benchmark

```bash
python -m benchmarks.benchmark_solvers --device cpu --classical-sizes 16 --quantum-sizes 2 --output benchmark-results/stage-1/cpu-v2.json
```

The direct script invocation remains supported. Each CLI case runs in a fresh
subprocess to make available allocator peaks local to that case. Inside that
process, measure one first call, one additional warm-up, and seven synchronized
warm calls. Report raw samples, median, minimum, maximum, iterations, objective,
and residual. The first call includes tracing/compilation and execution; it is
not pure compiler time. Python process startup is outside isolated solve timing.

The timed boundary is the host API, including its input validation and
transfers. Additional benchmark objective checks run outside the timer. Do not
retain every dense coupling while sampling, since that would inflate live
memory. All sampled solves must converge with finite outputs.

Read memory statistics from the devices owning the returned coupling.
`bytes_in_use`, `peak_bytes_in_use`, allocator reservation, and capacity are
separate nullable fields. CPU backends may expose no such measurements.
Direct calls to the benchmark functions use a shared process; their peaks are
explicitly process-wide and must not be reported as isolated per-case peaks.

## Comparison identity

Schema version 2 records the input digest, objective convention, shape, dtype,
geometry, epsilon, tolerance, iteration cap, output policy, selected devices,
host/CPU affinity, Python/JAX/jaxlib/NumPy/plugin versions, relevant runtime
environment, and source revision. Compare performance only for matching
scientific and environment contracts. Source revisions may differ so changes
can be evaluated; record both.

Flow records additionally need the energy, time step, physical horizon, inner
tolerance, and checkpoint/output policy. A different precision, objective,
geometry, tolerance, runtime, or allocation is a new experiment, not a faster
implementation of the same case. Nonconvergence, nonfinite measurements,
invalid timings, and legacy records missing comparison identity are rejected.

Small independent reference tests validate each family; they do not certify
every large problem's optimality. Report missing GPU model/driver/topology
details explicitly and acquire complete hardware characterization before
publishing GPU comparisons.

## Native and distributed decisions

Record actual usable memory, device links, CPU affinity, runtime/driver,
interconnect, and process mapping on each allocation. One/two devices are
initial test cases, not API limits. Keep solve latency, problem capacity, and
independent experiment throughput as separate measurements.

Use profiling and Amdahl's law before selecting a custom kernel. Retain a
native performance extension only with the agreed full-solve speed or memory
benefit at matched accuracy, or a documented capability gain. No CPU result
alone decides whether GPU spectral work needs native integration.
