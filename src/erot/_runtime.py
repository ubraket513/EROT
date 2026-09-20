"""JAX precision and device helpers with no import-time configuration."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .types import Precision


def enable_requested_precision(precision: Precision) -> None:
    """Enable JAX x64 when the caller explicitly requests float64."""

    if precision == "float64" and not jax.config.x64_enabled:
        jax.config.update("jax_enable_x64", True)


def real_dtype(precision: Precision) -> jnp.dtype:
    return jnp.dtype(jnp.float64 if precision == "float64" else jnp.float32)


def complex_dtype(precision: Precision) -> jnp.dtype:
    return jnp.dtype(jnp.complex128 if precision == "float64" else jnp.complex64)


def resolve_device(spec: str) -> jax.Device:
    """Resolve ``auto``, ``cpu``, ``gpu``, or ``cuda[:index]`` to a JAX device."""

    normalized = spec.strip().lower()
    if normalized == "auto":
        gpu_devices = (
            jax.devices("gpu")
            if any(device.platform == "gpu" for device in jax.devices())
            else []
        )
        return gpu_devices[0] if gpu_devices else jax.devices("cpu")[0]

    if normalized.startswith("cuda"):
        normalized = normalized.replace("cuda", "gpu", 1)

    platform, separator, raw_index = normalized.partition(":")
    if platform not in {"cpu", "gpu"}:
        raise ValueError("device must be 'auto', 'cpu', 'gpu', or 'cuda[:index]'")

    try:
        devices = jax.devices(platform)
    except RuntimeError as exc:
        raise ValueError(f"requested {platform!r} device is unavailable") from exc
    if not devices:
        raise ValueError(f"requested {platform!r} device is unavailable")

    index = int(raw_index) if separator else 0
    if index < 0 or index >= len(devices):
        raise ValueError(
            f"device index {index} is unavailable; found {len(devices)} {platform} device(s)"
        )
    return devices[index]
