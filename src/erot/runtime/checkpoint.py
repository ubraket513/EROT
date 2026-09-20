"""Atomic local checkpoint generations with explicit, non-pickle state encoding."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path

import jax
import numpy as np

_REQUIRED_METADATA = {"config_digest", "input_digest", "dtype", "topology"}
_GENERATION = re.compile(r"checkpoint-[0-9a-f]{32}")


def _registry():
    from ..flows.state import EntropicJKOState, JKODiagnostics, PDHGState
    from ..flows.trajectory import FlowState
    from ..solvers.quantum import QuantumDykstraState
    from ..solvers.quantum_entropy import QuantumEntropyDiagnostics, QuantumEntropyState
    from ..solvers.state import SinkhornState, SolverDiagnostics

    classes = (
        EntropicJKOState,
        JKODiagnostics,
        PDHGState,
        FlowState,
        QuantumDykstraState,
        QuantumEntropyDiagnostics,
        QuantumEntropyState,
        SinkhornState,
        SolverDiagnostics,
    )
    return {cls.__name__: cls for cls in classes}


def _metadata(value):
    if not isinstance(value, dict) or not _REQUIRED_METADATA <= value.keys():
        raise ValueError(
            "checkpoint metadata requires configuration, input, dtype and topology"
        )
    # Round-trip also copies nested containers and enforces JSON-only metadata.
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _encode(value, arrays, registry):
    if isinstance(value, (jax.Array, np.ndarray, np.generic)):
        array = np.asarray(jax.device_get(value))
        if array.dtype.kind not in "biufc":
            raise ValueError("checkpoint arrays must have numeric or boolean dtype")
        name = f"a{len(arrays)}"
        arrays[name] = array
        return {"kind": "array", "name": name}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        name = type(value).__name__
        if registry.get(name) is not type(value):
            raise ValueError("unsupported checkpoint state type")
        return {
            "kind": "namedtuple",
            "name": name,
            "items": [_encode(item, arrays, registry) for item in value],
        }
    if isinstance(value, (tuple, list)):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "items": [_encode(item, arrays, registry) for item in value],
        }
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("checkpoint dictionary keys must be strings")
        return {
            "kind": "dict",
            "items": {
                key: _encode(item, arrays, registry) for key, item in value.items()
            },
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("nonfinite scalar checkpoint values must be array leaves")
        return {"kind": "scalar", "value": value}
    raise ValueError(f"unsupported checkpoint value type: {type(value).__name__}")


def _decode(node, arrays, registry):
    kind = node["kind"]
    if kind == "array":
        return arrays[node["name"]]
    if kind == "scalar":
        value = node["value"]
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise ValueError("invalid scalar checkpoint node")
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("invalid scalar checkpoint value")
        return value
    if kind in ("tuple", "list", "namedtuple"):
        items = [_decode(item, arrays, registry) for item in node["items"]]
        if kind == "namedtuple":
            if node["name"] not in registry:
                raise ValueError("unknown checkpoint state type")
            cls = registry[node["name"]]
            if len(items) != len(cls._fields):
                raise ValueError("checkpoint state field count mismatch")
            return cls(*items)
        return tuple(items) if kind == "tuple" else items
    if kind == "dict":
        return {
            key: _decode(value, arrays, registry)
            for key, value in node["items"].items()
        }
    raise ValueError("unknown checkpoint tree kind")


def save_checkpoint(path: str | Path, state, metadata: dict) -> Path:
    """Durably write a generation, then atomically publish its LATEST pointer.

    Callers own the run directory and serialize writers. An interrupted save may
    leave an unpublished generation; it never mutates an earlier generation.
    Filesystem must support POSIX rename and fsync semantics.
    """
    metadata = _metadata(metadata)
    arrays = {}
    tree = _encode(state, arrays, _registry())
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    _fsync_directory(root.parent)
    token = uuid.uuid4().hex
    pending = root / f".pending-{token}"
    generation = root / f"checkpoint-{token}"
    pending.mkdir()
    payload = pending / "arrays.npz"
    with payload.open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    manifest = {
        "schema_version": 1,
        "metadata": metadata,
        "tree": tree,
        "arrays_sha256": _digest(payload),
        "arrays": {
            key: {"shape": list(value.shape), "dtype": value.dtype.str}
            for key, value in arrays.items()
        },
    }
    _write_json(pending / "manifest.json", manifest)
    _fsync_directory(pending)
    os.replace(pending, generation)
    _fsync_directory(root)
    pointer = root / f".latest-{token}"
    with pointer.open("x", encoding="utf-8") as stream:
        stream.write(generation.name + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pointer, root / "LATEST")
    _fsync_directory(root)
    return generation


def load_checkpoint(path: str | Path, expected_metadata: dict, *, generation=None):
    """Read the last published generation, returning NumPy-backed state and metadata.

    Compatibility is exact for every supplied expected metadata field. Array
    An explicit generation reads that immutable record instead of LATEST.
    shapes/dtypes are checked before state reconstruction; callers explicitly
    place the loaded leaves on their selected JAX device afterwards.
    """
    expected = _metadata(expected_metadata)
    root = Path(path)
    name = (
        (root / "LATEST").read_text(encoding="utf-8").strip()
        if generation is None
        else generation
    )
    if not isinstance(name, str) or not _GENERATION.fullmatch(name):
        raise ValueError("invalid checkpoint generation name")
    generation = root / name
    if generation.is_symlink() or generation.resolve().parent != root.resolve():
        raise ValueError("invalid checkpoint generation location")
    manifest_path, payload = generation / "manifest.json", generation / "arrays.npz"
    if manifest_path.is_symlink() or payload.is_symlink():
        raise ValueError("checkpoint files must not be symbolic links")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["schema_version"] != 1:
            raise ValueError("unsupported checkpoint schema")
        metadata = _metadata(manifest["metadata"])
        for key, value in expected.items():
            if key not in metadata or metadata[key] != value:
                raise ValueError(f"incompatible checkpoint metadata: {key}")
        if _digest(payload) != manifest["arrays_sha256"]:
            raise ValueError("checkpoint array checksum mismatch")
        with np.load(payload, allow_pickle=False) as archive:
            if set(archive.files) != set(manifest["arrays"]):
                raise ValueError("checkpoint array names mismatch")
            arrays = {}
            for key, spec in manifest["arrays"].items():
                array = archive[key]
                if (
                    array.dtype.kind not in "biufc"
                    or list(array.shape) != spec["shape"]
                    or array.dtype.str != spec["dtype"]
                ):
                    raise ValueError("checkpoint array shape/dtype mismatch")
                arrays[key] = array
        state = _decode(manifest["tree"], arrays, _registry())
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("malformed checkpoint manifest") from exc
    return state, metadata
