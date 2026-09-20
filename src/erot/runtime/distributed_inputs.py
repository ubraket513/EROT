"""Deterministic rank-local point clouds and addressable solver state."""

import hashlib
import json
from dataclasses import dataclass

import jax
import numpy as np
from jax.experimental import multihost_utils as mhu

from ..solvers.state import SinkhornState
from .distributed import global_from_local, partition_rows
from .distributed_checkpoint import collective_call


def assert_rank_agreement(value, label):
    """Compare small metadata digests without gathering numerical inputs."""
    digest = hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).digest()
    gathered = np.asarray(
        mhu.process_allgather(np.frombuffer(digest, np.uint8))
    ).reshape(-1, 32)
    if not (gathered == gathered[0]).all():
        raise ValueError(f"processes disagree on {label}")


def array_digest(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _points(seed, tag, indices, features, dtype):
    # Row-indexed streams make generation independent of process partitioning.
    return np.asarray(
        [
            np.random.default_rng(
                np.random.SeedSequence([seed, tag, int(index)])
            ).normal(size=features)
            for index in indices
        ],
        dtype=dtype,
    )


@dataclass
class DistributedInputs:
    x: object
    y: object
    a: object
    b: object
    initial: SinkhornState
    input_digest: str
    local_metadata: dict
    topology: dict


def create_inputs(config, mesh):
    rank = jax.process_index()
    devices = list(mesh.devices.flat)
    partitions = [
        partition_rows(config["n"], len(devices), index)
        for index, device in enumerate(devices)
        if device.process_index == rank
    ]

    def generate():
        indices = np.concatenate(
            [np.arange(p.start, p.start + p.size) for p in partitions]
        )
        valid = indices < config["n"]
        x = _points(config["seed"], 0, indices, config["features"], config["dtype"])
        x[~valid] = 0
        y = _points(
            config["seed"], 1, range(config["m"]), config["features"], config["dtype"]
        )
        a = np.where(valid, 1 / config["n"], 0).astype(config["dtype"])
        b = np.full(config["m"], 1 / config["m"], dtype=config["dtype"])
        return x, y, a, b

    x, y, a, b = collective_call("local input generation", generate)
    assert_rank_agreement(array_digest(y, b), "replicated target inputs")
    local_digest = array_digest(x, y, a, b)
    gathered = np.asarray(
        mhu.process_allgather(np.frombuffer(bytes.fromhex(local_digest), np.uint8))
    )
    input_digest = hashlib.sha256(gathered.tobytes()).hexdigest()
    f = np.where(a > 0, 0, -np.inf).astype(config["dtype"])
    g = np.zeros_like(b)
    initial = restore_state(SinkhornState((f, g), np.asarray(0, np.int32)), mesh)
    topology = {
        "processes": jax.process_count(),
        "devices": len(devices),
        "local_device_counts": [
            sum(d.process_index == p for d in devices)
            for p in range(jax.process_count())
        ],
        "platform": devices[0].platform,
        "device_kinds": [d.device_kind for d in devices],
        "global_source_shape": [
            sum(
                partition_rows(config["n"], len(devices), i).size
                for i in range(len(devices))
            ),
            config["features"],
        ],
        "target_shape": list(y.shape),
    }
    return DistributedInputs(
        global_from_local(x, mesh),
        global_from_local(y, mesh, True),
        global_from_local(a, mesh),
        global_from_local(b, mesh, True),
        initial,
        input_digest,
        {
            "local_input_digest": local_digest,
            "partitions": [list(p) for p in partitions],
        },
        topology,
    )


def local_scalar(value):
    """A replicated scalar read from one addressable shard, without a gather."""
    return np.asarray(value.addressable_shards[0].data)


def local_state(state):
    """Copy only this process's source potential shards and replicated state."""
    shards = sorted(
        state.potentials[0].addressable_shards,
        key=lambda shard: shard.index[0].start or 0,
    )
    f = np.concatenate([np.asarray(shard.data) for shard in shards])
    g = np.asarray(state.potentials[1].addressable_shards[0].data)
    return SinkhornState((f, g), local_scalar(state.iterations))


def restore_state(state, mesh):
    return SinkhornState(
        (
            global_from_local(state.potentials[0], mesh),
            global_from_local(state.potentials[1], mesh, True),
        ),
        global_from_local(state.iterations, mesh, True),
    )
