# Stage 6 distributed software verification

Historical integration record, captured before the subsequent repository cleanup.
See [the current repository map](repository-map.md) for the retained layout;
removed files and detailed task plans remain in EROT Git history.

The experimental row-partitioned implicit Sinkhorn path now has pure collective
state/diagnostics, pre-device process initialization, rank-local deterministic
inputs, complete distributed checkpoints, an installed driver and configurable
Slurm launch templates. Base APIs retain their single-device behavior. The tested
distributed runtime is Python 3.12.13/JAX 0.11.2/jaxlib 0.11.2; a CPU pin file is
included. Older base dependencies are not certified for distributed APIs.

Fresh two-virtual-device tests compare float32/64 objective, residual and gauge
to dense and blocked references, including uneven padding, zero target support,
a whole empty source shard, small epsilon, exact continuation, zero budget,
invalid inputs and counter overflow. JAX manual-axis checking remains enabled;
shared tile reductions explicitly mark local buffers only in distributed mode.
Compiler inspection found bounded tiles and approximately linear storage growth.

A real localhost two-process/Gloo test generates only local source rows and
matches the dense reference. It saves a three-sweep state, restores it and reaches
the same final potentials. Rank-local write failure preserves the old global
pointer; missing rank data rejects the load on all processes. Unit checks cover
exact rank-generation selection, checksums, interrupted publication, incompatible
metadata, schema, topology and incomplete manifests.

Driver tests cover uninterrupted versus stopped/resumed exact checkpoint arrays,
cross-rank configuration disagreement, global run ownership, changed restart
inputs, numerical exhaustion and one-process/two-device rank-script execution.
A malformed JSON type on only one rank initially caused a 60-second test timeout;
collective normalization now propagates that failure to both ranks. One-controller
and two-controller generation produce matching iteration counts and residuals.

Wheel and sdist build and artifact inspection passed. Outside the checkout, the
base-only installed erot-distributed command stopped the supplied 257-by-383 case
at 50 sweeps, then resumed on two virtual CPU devices to 107 sweeps and residual
8.89965e-9, with three checkpoint chunks. No optional optimizer/reference solver
was required. Slurm scripts pass shell syntax checks, and the rank script has a
local execution smoke; this is not a scheduler certification.

The [scaling protocol](../performance/distributed-transport.md) records sequential
one/two-virtual-device measurements and explains their limits. Warmup, compilation,
synchronized repetitions, compiler accounting, optional runtime memory counters
and rank-specific profiler capture are separate. CPU trace capture succeeded.
Opt-in actual two-GPU tests are prepared and skipped here. GPU speedup/capacity,
peak live memory, communication share, Slurm and interconnect gates remain open.
Distributed dense PDHG and quantum spectral solves are explicitly unsupported.

## Final verification and review

The final CPU suite passed 268 tests with 11 expected skips: nine unavailable
GPU checks and two opt-in artifact checks, which passed separately. Ruff and
formatting passed. Final wheel/sdist builds and both artifact inspections passed.
One fresh reviewer found a Slurm spooling path defect: resolving the rank helper
beside the copied batch script fails. The template now requires an explicit
readable shared absolute EROT_RANK_SCRIPT path. A copied-script regression first
failed with exit127 and now completes a real CPU solve through a local srun shim.
No material review findings or deferred minors remain.

Review rulings retain documented boundaries: actual GPU/Slurm/interconnect
promotion and filesystem power-loss certification await target access; process
death inside collectives is handled by JAX/job orchestration; resharded restart
and distributed PDHG/quantum spectral solves are not supported. Low-level callers
must supply consistent replicated arguments; the production driver checks their
configuration and generated-input identity. Violating that low-level contract
can produce divergence and is not a supported recovery mode.
