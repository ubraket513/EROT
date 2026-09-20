"""Explicit process initialization and row ownership for distributed solvers.

Importing this module does not initialize or import a numerical backend.
"""

from typing import NamedTuple


class RowPartition(NamedTuple):
    start: int
    size: int
    valid_size: int


def partition_rows(n: int, parts: int, index: int) -> RowPartition:
    """Equal padded extents; trailing rows outside n have zero source mass."""
    if any(type(value) is not int for value in (n, parts, index)) or not (
        n > 0 and parts > 0 and 0 <= index < parts
    ):
        raise ValueError("positive row/partition counts and a valid index are required")
    size = (n + parts - 1) // parts
    start = index * size
    return RowPartition(start, size, max(0, min(size, n - start)))


def initialize_runtime(
    coordinator_address=None, num_processes=1, process_id=0, local_device_ids=None
):
    """Initialize all processes before querying devices; no implicit discovery.

    Every process must call with the same world size/coordinator and its own ID.
    JAX's error for an already initialized backend is intentionally preserved.
    """
    if type(num_processes) is not int or num_processes < 1:
        raise ValueError("num_processes must be a positive integer")
    if type(process_id) is not int or not 0 <= process_id < num_processes:
        raise ValueError("process_id must be within the process world")
    if local_device_ids is not None and (
        not isinstance(local_device_ids, (tuple, list))
        or not local_device_ids
        or any(type(i) is not int or i < 0 for i in local_device_ids)
        or len(set(local_device_ids)) != len(local_device_ids)
    ):
        raise ValueError("local_device_ids must be distinct nonnegative integers")
    if num_processes == 1:
        if coordinator_address is not None or local_device_ids is not None:
            raise ValueError(
                "single-process mode uses externally configured device visibility"
            )
        return
    if not isinstance(coordinator_address, str) or not coordinator_address:
        raise ValueError("multiprocess mode requires an explicit coordinator_address")
    import jax

    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id,
        local_device_ids=local_device_ids,
    )


def make_row_mesh():
    """A process-major row mesh of all participating devices."""
    import jax
    import numpy as np
    from jax.sharding import Mesh

    if not hasattr(jax, "shard_map") or not hasattr(
        jax, "make_array_from_process_local_data"
    ):
        raise RuntimeError(
            "distributed mode requires current shard_map APIs; validated on JAX 0.11.2"
        )
    devices = sorted(
        jax.devices(), key=lambda device: (device.process_index, device.id)
    )
    if len({device.platform for device in devices}) != 1:
        raise ValueError("distributed row mesh requires a homogeneous device platform")
    return Mesh(np.asarray(devices), ("rows",))


def global_from_local(local_array, mesh, replicated=False):
    """Construct a global array from this process's addressable data only.

    Sharded input concatenates local row shards in mesh order. Replicated input
    must be identical on all ranks; this low-level constructor does not perform
    an expensive equality check. The distributed driver validates input identity.
    """
    import jax
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    if mesh.axis_names != ("rows",):
        raise ValueError("expected a one-dimensional rows mesh")
    return jax.make_array_from_process_local_data(
        NamedSharding(mesh, P() if replicated else P("rows")), local_array
    )
