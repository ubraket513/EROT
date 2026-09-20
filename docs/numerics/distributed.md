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
