# Experimental distributed classical transport

This module targets row-partitioned two-marginal squared-Euclidean Sinkhorn.
It is under implementation and is not promoted as validated GPU functionality.
The development CPU combination is Python 3.12.13, JAX/jaxlib 0.11.2 and the
pins in requirements/distributed-cpu.txt. The base package's older dependency
floor does not certify distributed APIs. Missing current shard_map support
raises an explicit error. GPU plugin/driver combinations require separate
hardware validation; the CPU pin file does not install a CUDA backend.

Initialize each process using initialize_runtime before any device query or
array creation. Multiprocess mode requires explicit coordinator_address,
num_processes and process_id; local_device_ids are optional explicit local
ordinals within scheduler visibility. Single-process mode uses the externally
configured visible devices. An already initialized backend is an error, not a
request to silently proceed without distributed initialization.

make_row_mesh builds a process-major one-dimensional mesh named rows.
global_from_local constructs global arrays from each process's local row data;
replicated target arrays must agree across all ranks. It does not gather source
geometry through the host. Equal padded extents allow n not divisible by the
number of devices. partition_rows reports global start, padded size and valid
size; callers must set padding source mass to zero. Local target data is
replicated deliberately, so target geometry and target potentials remain a
per-device capacity limit.

The runtime test forces two host CPU devices in a fresh subprocess and checks
global/local shapes, replication and rejection of late initialization. This
proves array-construction behavior on the stated CPU runtime, not CUDA peer
communication, NCCL behavior or multi-node scalability.

Implementation references are the official [JAX multiprocess guide](https://docs.jax.dev/en/latest/multi_process.html)
and [shard_map guide](https://docs.jax.dev/en/latest/notebooks/shard_map.html).

## Collective solver contract

make_distributed_sinkhorn(mesh, block_size=128) returns a compiled function with
signature solve(x,y,a,b,epsilon,tolerance,max_iterations,state). x and a have
row ownership; y and b are replicated. Supply a SinkhornState with row-sharded f,
replicated g and a replicated scalar iteration count. Initialize positive-support
potentials to zero and zero-support potentials to negative infinity. Padded
coordinates must be finite and their source masses zero. max_iterations is
additional work on an unchanged problem. Geometry is squared Euclidean; arbitrary
cost matrices and multi-marginal distributions retain their separate dense APIs.

Row updates use complete target blocks without communication. Column updates
combine local log-sums using a global maximum followed by a global sum of
exponentials shifted by that maximum. All-empty supports use a safe zero shift
and remain negative infinity. Residuals are the maximum of global row L1 error
and column L1 error; no coupling is gathered. The source-mass-weighted gauge is
global. Rank-local input validity is reduced before the convergence loop, so
status, cumulative iterations and termination are consistent across the mesh.
JAX manual-axis checking stays enabled; loop buffers explicitly declare their
local variation rather than disabling the checker.

Each completed sweep performs four target-length collective reductions (max/sum
for the column update and max/sum for the column residual), plus two scalar sums
for the gauge and row residual. Initialization additionally reduces global mass,
validity and the initial residual. This counts logical collective payloads, not
network bytes or measured communication time: implementation, rank count and
interconnect determine actual traffic. Replicated targets cost O(m*features+m)
per device. Source data/potentials cost O(n*features/devices+n/devices), and
intermediate cost tiles are bounded by block_size. Arithmetic remains quadratic.

The compiled function supports .lower(...).compile() for buffer inspection and
synchronized timing. Dynamic convergence does not advertise reverse-mode
through-solve differentiation. Tests compare float32 and float64 solutions,
objectives and residuals to dense/blocked references; include uneven padding,
zero support on a whole shard, small epsilon, exact continuation and global
invalid-input status. These are virtual CPU device checks only.

A two-virtual-CPU compiler inspection at tile size 3 reported 38,068 bytes for
padded n=258,m=383 and 74,380 bytes for n=514,m=766. No full global or local
coupling shapes appeared in the compiled representation. This is compiler
buffer accounting, not measured live GPU memory or allocator reservation.

## Distributed checkpoint publication

All processes call save_distributed_checkpoint with their local addressable
state and compatibility metadata. The owning driver must hold the global run
lock on process zero. Each save chooses a shared unique generation, writes an
immutable rank checkpoint on every process, and exchanges success before
publication. Process zero writes a manifest naming each exact rank generation
and its manifest checksum, then atomically replaces the global LATEST pointer.
An incomplete rank write leaves the previous globally published checkpoint
selected. Orphan generations are ignored, not automatically deleted.

All processes also participate in load_distributed_checkpoint. They read the
same global manifest and their own rank payload, validate topology and metadata,
and exchange load success before proceeding. A missing/corrupt rank makes the
load fail globally. Local rank LATEST pointers are not authoritative once the
global manifest is published. No full source vector or transport plan is
assembled on the host. Same-topology restart is supported; resharded restart is
not advertised. Replicated target state is intentionally stored per process.

The filesystem must provide the documented POSIX locking/rename/fsync semantics
and shared visibility. Python I/O failures are coordinated; a process killed
inside a collective is handled by the distributed runtime and job launcher,
not by silently proceeding with fewer ranks. Target-filesystem power-loss
behavior and actual scheduler failure recovery remain hardware/environment gates.

A localhost two-process CPU test supplies four rows per process for seven valid
source points, matches the single-process dense solution, saves/restores a
three-sweep state and reaches the same final potentials. It also simulates a
write failure on only one rank and removes one rank's payload: publication and
load respectively fail on all participants, preserving the old global pointer
in the interrupted-save case. These tests use Gloo CPU collectives; they do not
validate CUDA/NCCL communication.

## Installed driver and local input generation

    erot-distributed experiments/configs/distributed-classical.json --run-directory results/distributed --device cpu

The CLI selects the platform explicitly with --device (default cpu), overriding
the configuration device field before identity calculation. Multiprocess launch
adds --coordinator host:port --processes N --process-id rank on every process;
initialize_runtime is called before device access. See [Slurm launch notes](../../hpc/README.md).

The driver accepts classical/blocked configurations from the shared schema. Its
workload generator is distinct from the independent-experiment generator:
source and target points use NumPy SeedSequence streams indexed by seed, cloud
and global row index, and marginals are uniform over valid rows. Each process
generates only its local padded source rows; targets are replicated. A change
in partitioning does not change valid input points. Generator identity is
recorded in checkpoint metadata. Input digests combine rank-local array digests;
only these small digests are gathered, not source arrays or couplings.

All ranks agree on configuration, operational controls, source/runtime identity,
replicated target identity and restored progress before entering the solve.
Rank zero holds the global run lock. Changed input, precision, runtime/source or
topology is rejected on resume. Source revision is observational; exact package
source content is the compatibility criterion. A deliberately stopped run is
checkpointed, numerical exhaustion/failure is failed with a nonzero exit, and
only convergence is completed. Failed states remain available for inspection.

Each chunk synchronizes all participating device work before checkpointing.
timing.jsonl reports the maximum observed rank duration for each chunk and
checkpoint phase; first-call compilation is combined with execution, subsequent
calls reuse the same dynamic-budget signature. result.json reports residual,
iterations, status, topology and setup/session timings. resources-rank-*.json
records each process's observed allocation. Scientific output does not include
a gathered transport plan. Shared-filesystem visibility and process launch
remain site responsibilities; Python configuration/I/O errors are coordinated
before the next collective phase.
