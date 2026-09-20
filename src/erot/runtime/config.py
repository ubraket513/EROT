"""Strict schema-v1 experiment configuration and stable run identities."""

import hashlib
import json
import math
import re

_DEFAULTS = {
    "schema_version": 1,
    "name": "experiment",
    "kind": "flow",
    "backend": "blocked",
    "energy": "entropy",
    "n": 16,
    "m": None,
    "features": 1,
    "seed": 42,
    "dtype": "float64",
    "device": "cpu",
    "epsilon": 0.2,
    "time_step": 0.02,
    "tolerance": 1e-6,
    "max_iterations": 10000,
    "inner_tolerance": 1e-10,
    "inner_iterations": 10000,
    "steps": 3,
    "chunk_size": 1,
    "block_size": 32,
    "snapshot_stride": None,
}


def normalize_config(config: dict) -> dict:
    """Validate all controls and return a JSON-compatible canonical configuration."""
    if not isinstance(config, dict) or set(config) - set(_DEFAULTS):
        raise ValueError("unknown experiment configuration fields")
    result = _DEFAULTS | config
    if type(result["schema_version"]) is not int or result["schema_version"] != 1:
        raise ValueError("unsupported experiment schema")
    if not isinstance(result["name"], str) or not re.fullmatch(
        r"[A-Za-z0-9_-]{1,64}", result["name"]
    ):
        raise ValueError(
            "name must contain 1-64 letters, digits, underscores or hyphens"
        )
    if result["kind"] not in ("flow", "classical", "quantum"):
        raise ValueError("kind must be flow, classical or quantum")
    if result["m"] is None:
        result["m"] = result["n"]
    for key in (
        "n",
        "m",
        "features",
        "seed",
        "max_iterations",
        "inner_iterations",
        "steps",
        "chunk_size",
        "block_size",
    ):
        value = result[key]
        minimum = 0 if key in ("seed", "inner_iterations") else 1
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not minimum <= value <= 2**31 - 1
        ):
            raise ValueError(f"{key} must be an integer in [{minimum}, 2**31-1]")
    stride = result["snapshot_stride"]
    if stride is not None and (
        isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0
    ):
        raise ValueError("snapshot_stride must be a positive integer or null")
    for key in ("epsilon", "time_step", "tolerance", "inner_tolerance"):
        value = result[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{key} must be finite and positive")
        result[key] = float(value)
    if result["dtype"] not in ("float32", "float64") or result["device"] not in (
        "cpu",
        "gpu",
    ):
        raise ValueError(
            "dtype/device must select supported explicit precision and platform"
        )
    allowed = {
        "flow": ("pdhg", "sinkhorn", "blocked"),
        "classical": ("sinkhorn", "blocked"),
        "quantum": ("entropy", "quadratic"),
    }
    if result["backend"] not in allowed[result["kind"]]:
        raise ValueError("backend is incompatible with workload kind")
    if result["energy"] not in ("entropy", "quadratic"):
        raise ValueError("energy must be entropy or quadratic")
    if result["kind"] == "flow":
        if result["n"] != result["m"]:
            raise ValueError("flow grid must have equal source/target sizes")
        if (
            result["backend"] != "pdhg"
            and result["inner_tolerance"] > result["tolerance"]
        ):
            raise ValueError("inner tolerance must not exceed outer tolerance")
    return result


def config_digest(config: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            normalize_config(config),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def run_identity(config: dict) -> str:
    config = normalize_config(config)
    return config["name"] + "-" + config_digest(config)[:16]
