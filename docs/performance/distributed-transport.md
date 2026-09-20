# Distributed classical scaling protocol

The experimental backend implements row-partitioned squared-Euclidean Sinkhorn.
It has virtual-device and localhost multi-process CPU correctness evidence.
Actual Hopper-or-newer GPU speed, capacity, peak live memory, CUDA collectives
and Slurm/interconnect behavior remain unvalidated. Multi-node promotion is
therefore gated. No GPU speedup target is claimed achieved.

## Reproducible benchmark

Each invocation uses fresh output and records configuration, a topology-independent
scientific case digest, software/source identity, per-rank resources, global/local
shapes and numerical status. Inputs use the driver's row-indexed generator and
uniform marginals. Compare equal case digests, tolerances and precision, and
inspect convergence before comparing times. Checkpointing is outside this
kernel/full-solve benchmark; use erot-distributed for checkpointed end-to-end jobs.

```bash
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=1 \
  python benchmarks/distributed_sinkhorn.py experiments/configs/distributed-classical.json \
  --output-directory benchmark-results/distributed-one --device cpu --repeats 5
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 \
  python benchmarks/distributed_sinkhorn.py experiments/configs/distributed-classical.json \
  --output-directory benchmark-results/distributed-two --device cpu --repeats 5
```

Compilation and the first synchronized execution are reported separately from
repeated full solves initialized at zero potentials. Each sample begins with a
process barrier and ends with block_until_ready; the reported duration is the
maximum rank time. Failed/inaccurate solves leave a report with valid=false and
return nonzero. Existing output directories are rejected to prevent mixing runs.

The supplied 257-by-383 float64 case, epsilon 0.3, tolerance 1e-8, tile 32, seed 42
converged in 107 sweeps on both CPU configurations. Five warm samples gave:

| Virtual CPU devices | Compilation seconds | Warm median seconds | Residual | Compiler bytes |
|---|---:|---:|---:|---:|
| 1 | 0.495 | 0.1852 | 8.90e-9 | 58,892 |
| 2 | 0.562 | 0.1385 | 8.90e-9 | 49,524 |

Both used the same eight-core process affinity, Python 3.12.13 and JAX/jaxlib 0.11.2.
Trials ran sequentially without concurrent validation jobs. Virtual CPU devices
share host resources; this is not a physical multi-GPU scaling experiment.
The ignored raw records are benchmark-results/stage6/{one-device,two-devices}/.
No full local or global coupling shapes appeared in the inspected executables.

## GPU and process validation

On an allocated CUDA environment, first run the opt-in numerical checks with two
visible GPUs and both CUDA/CPU backends available:

    JAX_PLATFORMS=cuda,cpu python -m pytest tests/gpu/test_distributed_gpu.py -q

The test chooses two visible devices, checks float32/64 against an independent
NumPy Sinkhorn reference, exercises an all-zero source shard and continuation,
and skips explicitly if fewer than two GPUs are available. A skip is not a pass.
Run the hardware inventory from the baseline protocol and retain exact package,
driver, device and allocation metadata.

For one-controller scaling, launch the benchmark above with --device gpu,
first with one allocated device visible and then with two. Preserve the site's
opaque CUDA_VISIBLE_DEVICES identifiers; select only within the allocation.
Keep problem controls and total CPU budget fixed, use a new output directory for
each run, and increase sizes only after convergence and memory checks pass.
A newer GPU is another configuration to validate, not an automatic fast path.

For two processes, launch the same benchmark command on both ranks with
--coordinator host:port --processes 2 --process-id rank --local-device-ids 0.
Each rank owns one GPU in its task-local visible allocation and shares the same
output path. The coordinator and CPU/GPU bindings come from site configuration,
as in hpc/distributed_rank.sh. CPU/Gloo two-process execution of this benchmark
is tested locally; CUDA/NCCL and actual Slurm launch are not. Compare this process
model's numerical results to the one-controller result before assessing timing.
Only promote to two nodes after the site's interconnect and launch checks pass.

## Memory and communication interpretation

Compiler memory is argument+output+temporary-alias buffer accounting. It excludes
compilation and allocator reservation and is not live GPU peak memory. Per-device
runtime bytes_in_use, peak_bytes_in_use, bytes_reserved and bytes_limit are
recorded when the backend supplies them; unavailable values are null. High-water
counters cover the whole process including setup/compilation/earlier calls. Do
not subtract them to infer an isolated workspace or claim they are a profiler's
live solve peak. Use fresh processes per size and inspect a memory timeline for
peak-live claims. Report per-device values, not only their sum.

Add --trace for local rank-specific profiler directories. The benchmark captures
synchronized warm solves without opening an external viewer or uploading traces.
A CPU capture smoke produced trace artifacts successfully. Follow the official
[JAX profiling guide](https://docs.jax.dev/en/latest/profiling.html) to inspect
these files with XProf/TensorBoard or a local trace viewer. For GPU investigations,
identify collective kernels and their overlap with compute on the critical path;
summing overlapping kernel durations is not elapsed communication time.
communication_share remains null until an actual trace analysis establishes it.
Logical payload counts in the numerical guide are not measured bandwidth.

Source geometry and f are row-sharded. Targets, b and g remain replicated, so
large target clouds still set a per-device capacity limit. Tiles remove the full
coupling representation but do not remove quadratic arithmetic. Benchmark speed
on cases fitting one device and capacity on larger cases separately; missing
one-device capacity is not a numeric speedup measurement.

## Separate PDHG and quantum decisions

Dense JKO PDHG stores a coupling and an extrapolated coupling, plus marginal and
dual vectors. Two n-by-m floating arrays alone cost 2*n*m*itemsize bytes before
cost and working buffers. Row partitioning could split this state and require
column-residual reductions, but it would retain quadratic storage. This is a
separate algorithm/communication implementation; the implicit Sinkhorn evidence
does not validate it. The current release keeps dense PDHG on one device and
supports many independent PDHG experiments through the worker launcher. A
single distributed PDHG solve is not advertised.

Quantum quadratic Dykstra stores a dense coupling and three correction matrices;
the dense cost and full spectral intermediates add further storage. Entropy QOT
also uses a full dense Gibbs spectral decomposition. Classical row sharding does
not implement a global PSD projection or distributed eigensolver. The selected
quantum capability remains the tested dense single-device path, with no measured
Hopper capacity limit yet. See [quantum feasibility](quantum-feasibility.md) for
projection/full-solve tools and the native distributed spectral decision gate.
Lanczos eigenpairs are not a replacement for the full positive spectral part.
