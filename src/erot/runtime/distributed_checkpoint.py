"""Collective publication of immutable rank-local checkpoint generations."""

import json
import os
import re
import uuid
from pathlib import Path

import jax
import numpy as np
from jax.experimental import multihost_utils as mhu

from .checkpoint import (
    _GENERATION,
    _digest,
    _fsync_directory,
    _write_json,
    load_checkpoint,
    save_checkpoint,
)

_GLOBAL_GENERATION = re.compile(r"distributed-[0-9a-f]{32}")


def collective_call(label, action):
    """Run local host work and exchange success before the next collective phase.

    Every rank must call in the same order. Process death is handled by the
    distributed runtime/job launcher; Python exceptions are propagated to all.
    """
    result, error = None, None
    try:
        result = action()
    except Exception as exc:
        error = exc
    success = mhu.process_allgather(np.asarray(error is None, np.int32))
    if not np.asarray(success).all():
        detail = str(error) if error is not None else "another rank failed"
        raise RuntimeError(f"{label}: {detail}") from error
    return result


def save_distributed_checkpoint(path, local_state, metadata):
    """Publish only after every rank has durably saved its complete local state.

    All ranks call with a shared POSIX path. The driver holds global ownership
    on rank0 for the entire run. Only local, addressable arrays may be passed.
    """
    root = Path(path)
    rank, processes = jax.process_index(), jax.process_count()
    token = np.frombuffer(
        uuid.uuid4().hex.encode() if rank == 0 else b"0" * 32, np.uint8
    )
    token = bytes(np.asarray(mhu.broadcast_one_to_all(token))).decode()
    generation = root / f"distributed-{token}"

    def create():
        if rank == 0:
            root.mkdir(parents=True, exist_ok=True)
            _fsync_directory(root.parent)
            generation.mkdir()
            _fsync_directory(root)

    collective_call("distributed checkpoint directory", create)
    collective_call(
        "distributed rank save",
        lambda: save_checkpoint(generation / f"rank-{rank:06d}", local_state, metadata),
    )

    def publish():
        if rank != 0:
            return
        records = []
        for index in range(processes):
            rank_root = generation / f"rank-{index:06d}"
            name = (rank_root / "LATEST").read_text().strip()
            if not _GENERATION.fullmatch(name):
                raise ValueError("invalid rank generation")
            records.append(
                {
                    "generation": name,
                    "manifest_sha256": _digest(rank_root / name / "manifest.json"),
                }
            )
        _write_json(
            generation / "manifest.json",
            {"schema_version": 1, "processes": processes, "ranks": records},
        )
        _fsync_directory(generation)
        pointer = root / f".latest-{token}"
        with pointer.open("x", encoding="utf-8") as stream:
            stream.write(generation.name + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pointer, root / "LATEST")
        _fsync_directory(root)

    collective_call("distributed checkpoint publication", publish)
    return generation


def load_distributed_checkpoint(path, expected_metadata):
    """Load this rank from one published global manifest and agree on success."""
    root = Path(path)
    rank, processes = jax.process_index(), jax.process_count()

    def read():
        name = (root / "LATEST").read_text().strip()
        if not _GLOBAL_GENERATION.fullmatch(name):
            raise ValueError("invalid distributed generation name")
        generation = root / name
        manifest_path = generation / "manifest.json"
        if generation.is_symlink() or manifest_path.is_symlink():
            raise ValueError("distributed checkpoint symbolic links are unsupported")
        manifest = json.loads(manifest_path.read_text())
        if (
            type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1
        ):
            raise ValueError("unsupported distributed checkpoint schema")
        if (
            type(manifest.get("processes")) is not int
            or manifest["processes"] != processes
        ):
            raise ValueError("incompatible distributed checkpoint topology")
        records = manifest.get("ranks")
        if not isinstance(records, list) or len(records) != processes:
            raise ValueError("incomplete distributed rank manifest")
        record = records[rank]
        local_name = record["generation"]
        if not isinstance(local_name, str) or not _GENERATION.fullmatch(local_name):
            raise ValueError("invalid rank generation")
        rank_root = generation / f"rank-{rank:06d}"
        local_manifest = rank_root / local_name / "manifest.json"
        if (
            rank_root.is_symlink()
            or (rank_root / local_name).is_symlink()
            or local_manifest.is_symlink()
        ):
            raise ValueError("rank checkpoint symbolic links are unsupported")
        if _digest(local_manifest) != record["manifest_sha256"]:
            raise ValueError("rank manifest checksum mismatch")
        return load_checkpoint(rank_root, expected_metadata, generation=local_name)

    return collective_call("distributed checkpoint load", read)
