# Blocked point-cloud transport

PointCloudGeometry stores two arrays of points. Its cost is squared Euclidean
distance, evaluated only for the requested tile. Direct point differences avoid
cancellation from the norm-expansion identity for large common coordinate
offsets. DenseGeometry adapts an existing cost matrix to the same interface;
it does not remove that matrix's storage.

    from functools import partial
    import jax
    from erot.geometry import PointCloudGeometry
    from erot.solvers.blocked_sinkhorn import solve_blocked_sinkhorn

    geometry = PointCloudGeometry(x, y)
    state, diagnostics = jax.jit(
        partial(solve_blocked_sinkhorn, block_size=64)
    )(geometry, (a, b), epsilon, tolerance, max_iterations)

All coordinates must be real floating arrays; marginals are nonnegative with
the same positive mass. Geometry shapes and block size are static compilation
parameters. The solver uses geometry precision and cost-unit potentials, with
the same weighted gauge and marginal L1 diagnostic as dense Sinkhorn. Enable
JAX x64 explicitly for tight tolerances. Additional-work state and separate
warm-start semantics match the dense core.

Streamed log-sum-exp accumulates each direction across tiles. Padding and
inactive supports contribute no mass, including completely masked tiles.
Residuals are streamed as well. Computational work remains quadratic in point
counts; tiling reduces storage rather than the number of pair interactions.
Coordinates whose differences or squared distances overflow require rescaling;
the solver does not promise arbitrary-scale arithmetic.

## Outputs and storage

No dense plan is returned by the blocked solver. Use erot.geometry.plan:

- plan_block returns a requested fixed-size coupling tile, padded with zeros.
- apply_transport computes pi @ vector or pi @ feature_matrix by tiles.
- transport_objective streams the same Shannon objective as dense Sinkhorn.

For transpose application, transpose geometry and reverse the potential tuple.
Output functions assume valid potentials and positive epsilon. They preserve a
common floating precision for accumulators. A complete SinkhornState is the
resume checkpoint; a few plan blocks are not.

With n and m points, d features and tile side B, point-cloud solver storage is
O((n+m)*d+n+m+B*B*d); the direct-difference temporary includes the feature
dimension. Applying the plan to k features additionally stores its (n,k) output
and bounded tile feature buffers. Compiler optimizations may fuse intermediates,
but the contract does not rely on fusion to avoid n*m storage. Full dense and
multi-marginal APIs retain their original storage limits.

## Reproducible measurements

    JAX_PLATFORMS=cpu python benchmarks/benchmark_blocked_sinkhorn.py \
      --n 257 --m 383 --features 3 --block-size 32 --compare-dense \
      --output benchmark-results/blocked-cpu.json

Use a separate process for each shape/block/device configuration. Avoid other
compute jobs when interpreting timing. The command records source revision,
input digest, random seed, runtime/device metadata, objective and residual,
compile time, first execution and seven synchronized warm samples. Dense
comparison uses the same points, marginals, epsilon, dtype and tolerance.

The dedicated erot.blocked-benchmark.v1 report is separate from the legacy
solver benchmark schema. It records matched accuracy before permitting a
successful comparison exit. Default outputs for both compared solvers are
potentials and diagnostics; objective evaluation is outside solve timing.

Compiled memory analysis reports argument, output, temporary and alias bytes.
The executable text is also inspected for full n*m or n-by-m shapes. These are
compiler buffer estimates, not runtime GPU peaks or total process memory.
Runtime allocator statistics are retained separately when available; their
peaks are cumulative for the process and can include other executables.
Linux peak RSS includes compiler and host allocations. When --compare-dense is
used, the final RSS and later allocator peaks also include the dense reference.

For GPU validation, run the same command with --device gpu and a CUDA-enabled
JAX environment (do not force JAX_PLATFORMS=cpu). It fails if GPU placement is
unavailable. Sweep block sizes and larger n/m under the site's resource budget.
For isolated blocked capacity measurement omit --compare-dense; otherwise
constructing the dense reference defeats the memory advantage. CPU/GPU
agreement tests live under tests/gpu; no GPU timing or measured peak claim
is established by the CPU checks.

## CPU evidence from this workspace

The following runs used float64, seed 42, three features, epsilon 0.3,
tolerance 1e-8 and tile side 32, with seven warm samples. Each final benchmark
ran in its own process after the test suite finished. Earlier concurrent runs
were replaced because their timings could include test/benchmark contention.

| Points n x m | Blocked compiler bytes | Dense compiler bytes | Blocked warm median | Dense warm median | Sweeps |
|---|---:|---:|---:|---:|---:|
| 257 x 383 | 53,492 | 2,479,028 | 0.2241 s | 0.07255 s | 113 |
| 514 x 766 | 95,092 | 9,879,132 | 0.7589 s | 0.1643 s | 107 |

Both paths converged at the same sweep counts. Maximum marginal L1 residuals
were 9.52e-9 and 9.29e-9; objective differences were zero and 2.22e-15.
Doubling both counts increased blocked compiler accounting by 1.78x versus
3.99x for dense. Compiled blocked executables contained no full n-by-m or
flattened n*m buffer shape. On these small CPU cases, dense execution was faster;
the demonstrated gain is storage, not runtime speed.

Combined blocked-plus-dense process peak RSS was 305,256 and 327,828 KiB,
including compilation and host/reference allocations; it is not a solver live
buffer measurement. CPU device allocator peak fields were unavailable. Actual
GPU peaks and tile choices remain unmeasured. Compiler accounting follows JAX's
[AOT memory-analysis contract](https://docs.jax.dev/en/latest/aot.html) and is
reported separately from runtime profiling.

Nonfinite numerical values in failed benchmark reports are JSON null, with the
solver's failure status retained. Such reports cannot become successful matched
comparisons. Re-run the recorded command/configuration on target hardware;
these CPU observations do not establish Hopper performance.

Entropic gradient-flow chunks accept the same geometry in their cost position
with transport_block_size. Their transport objective and dual mass term also
stream tiles; compiled-flow tests check the absence of full cost/coupling
shapes. See the [flow interface](../numerics/gradient-flows.md).
PDHG remains a dense coupling method and is not accelerated by this interface.
