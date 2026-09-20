# Quantum backend feasibility and validation protocol

Decision: keep JAX as the supported implementation while collecting the GPU
measurements that could justify a native backend. C++/CUDA is a possible
capability extension, not an assumed speed improvement. Fortran is not required
by the current algorithms. No Hopper-or-newer allocation is available in this
workspace; the user requested runnable validation tools in its place.

## What can be reproduced now

```bash
python -m benchmarks.hardware_inventory --backend cpu --output benchmark-results/hardware.json
python -m benchmarks.benchmark_quantum_projection --device cpu --dimension 16 \
  --dtype complex128 --output benchmark-results/projection.json
```

The projection benchmark times full `eigh` and PSD projection separately, with a
first invocation, another warm-up, and seven synchronized measurements each.
Host generation, transfer, and full independent NumPy complex128 reference
validation are outside warm device timing. Both complex64 and complex128 have
explicit accuracy thresholds. Each CLI invocation handles one size; use a fresh
process per size. Per-operation memory snapshots share a process high-water
mark and must not be subtracted or described as isolated workspace peaks.

CPU dimension 16, complex128, seed 17: projection relative error
4.42e-16, Hermiticity error zero, minimum eigenvalue -3.38e-16. CPU smoke tests
also validate complex64 against the double-precision reference. These are
correctness observations, not GPU speed or capacity estimates. The GPU inventory
command writes an unavailable-backend report and exits nonzero in the current
environment. No silent CPU result is accepted as GPU validation.

## Procedure on a target allocation

Replace `--backend cpu` with `--backend gpu` for inventory and `--device cpu`
with `--device gpu` for projection. The installed environment must supply the
appropriate CUDA-enabled JAX extra. Record its exact package versions with
`python -m pip freeze` alongside each result. Run inventory inside the actual
allocation, separately for each host/process mapping. NVIDIA XML and topology
output preserve driver, visible memory, MIG/sharing details and device links;
CPU affinity, Slurm fields, host memory and cgroup limits are recorded separately.
Absent utilities and failed commands are explicit outcomes. Inventory is not a
collective-bandwidth measurement or a GPU support certification.

Begin with subsystem dimensions 4 and 8 (matrix dimensions 16 and 64), in both
complex precisions. Check accuracy before increasing to subsystem dimensions
16 and 32 (matrix dimensions 256 and 1024). The full NumPy reference also
consumes host memory/time; budget it explicitly. A single projection is not a
full OT solve: run the existing quantum solver benchmark separately and retain
its convergence diagnostics. Profile one GPU first. Two GPUs require a genuine
distributed implementation and verified communication; visibility alone does
not pool their memory. Additional device counts are explicit experiments.

For `N=n*m`, one complex128 matrix uses `16*N*N` bytes. At `n=m=128`, that is
4 GiB. Cost, coupling and three Dykstra corrections already account for five
such matrices. Projection intermediates, eigenvectors, allocator retention and
backend workspace add more. `memory_budget` exposes this accounting with
unknown workspace and measured peak fields left null; it is not a safe upper
bound. Establish headroom on a smaller real run before escalating sizes.

## Native boundary and unresolved measurements

A candidate native interface accepts GPU-resident Hermitian tiles and returns
an eigendecomposition or PSD-projected tiles with explicit status, precision,
layout and communicator ownership. Prefer a narrow FFI operation only if layout
conversion and synchronization do not dominate repeated projections. If every
iteration otherwise redistributes the full matrix, evaluate a coarse-grained
native quantum loop that retains the distributed layout. Preserve the dense
JAX reference and compare the same primal objective and residuals.

The current NVIDIA cuSOLVERMp API documents real and complex single/double
precision `Syevd`, square aligned block sizes, full-matrix starts, and separate
host/device workspace queries. Query the installed release's buffer sizes;
never reuse constants from another release. See the
[API reference](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html).

Local matrices are column-major and distributed through a 2D block-cyclic
layout. The current initialization path uses NCCL communicators; older CAL/MPI
examples are not an interchangeable ABI. Pin the release and dependencies
before implementing a bridge. See
[data layout](https://docs.nvidia.com/cuda/cusolvermp/getting_started/index.html)
and [NCCL initialization](https://docs.nvidia.com/cuda/cusolvermp/usage/initialization/nccl.html).
The live [release notes](https://docs.nvidia.com/cuda/cusolvermp/release_notes/index.html)
were checked on 2026-09-21; cached Context7 examples still mention the older
power-of-two block restriction, which the live notes report as fixed.

The next resolving experiment is a two-GPU complex128 distributed
`eigh`/reconstruction round trip with residual, orthogonality, layout-conversion
time and peak workspace recorded, followed by repeated PSD projections.
No native speedup threshold, supported multi-GPU quantum size or future NVIDIA
architecture certification can be established from the current CPU evidence.
Pallas/custom CUDA kernels remain candidates for measured bottlenecks; neither
replaces the full eigensolver merely by compiling Python at a lower level.
