# Development roadmap

The integrated 0.2 implementation includes classical transport, gradient flows,
dense quantum entropy transport, blocked geometry, reproducible experiments and
experimental distributed Sinkhorn. See the [support matrix](../support-matrix.md)
and [release verification](stage-8-validation.md) for evidence and limitations.
Completed implementation task lists live in Git history before the repository
cleanup; they are not required to build, use or maintain the package.

## Hardware validation

Hopper-or-newer GPU and real Slurm access was unavailable during implementation.
The prepared GPU tests, profiler/benchmark tools and configurable launch scripts
still need actual hardware validation: single GPU, two GPU controllers/processes,
isolated peak live memory, tile tuning, throughput, communication and restart.
Multi-node promotion additionally requires interconnect and filesystem evidence.
See [HPC instructions](../../hpc/README.md) and the
[performance protocol](../performance/baseline-protocol.md).

## Conditional research extensions

- Native acceleration only after a measured bottleneck or required capability
  passes the [admission criteria](../performance/native-decision.md).
- Distributed quantum spectral work, structured/chordal formulations and low-rank
  approximations require their own mathematical and performance evidence.
- Generalized KL-proximal JKO, dynamic density/flux PDHG and end-to-end solver
  differentiation need separately validated numerical contracts.

These are research directions, not capabilities advertised by the current release.
