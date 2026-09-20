"""Complete-state checkpoint integrity and interrupted publication."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from erot.flows import initialize_flow
from erot.runtime.checkpoint import load_checkpoint, save_checkpoint
from erot.solvers.quantum import solve_quantum_quadratic
from erot.solvers.quantum_entropy import solve_quantum_entropy
from erot.solvers.state import SinkhornState

jax.config.update("jax_enable_x64", True)


def metadata():
    return {
        "config_digest": "config",
        "input_digest": "input",
        "dtype": "float64",
        "topology": {"processes": 1, "devices": 1},
        "runtime": {"jax": jax.__version__},
        "source_digest": "source",
        "seed": 7,
        "rng_state": None,
    }


@pytest.mark.parametrize("kind", ["sinkhorn", "pdhg", "flow", "quadratic", "entropy"])
def test_complete_state_round_trip(tmp_path, kind):
    if kind == "sinkhorn":
        state = SinkhornState(
            (jnp.array([0.0, -jnp.inf]), jnp.array([1.0, 2.0])), jnp.int32(7)
        )
    elif kind in ("pdhg", "flow"):
        state = initialize_flow(
            jnp.array([0.3, 0.7]), backend="pdhg" if kind == "pdhg" else "sinkhorn"
        )
    else:
        a = jnp.array([[0.6, 0.1j], [-0.1j, 0.4]])
        fn = solve_quantum_quadratic if kind == "quadratic" else solve_quantum_entropy
        state, _ = fn(jnp.zeros((4, 4)), a, a, 0.5, 1e-8, 2)
    generation = save_checkpoint(tmp_path, state, metadata())
    restored, actual_metadata = load_checkpoint(tmp_path, metadata())
    assert type(restored) is type(state)
    assert actual_metadata == metadata()
    for actual, expected in zip(jax.tree.leaves(restored), jax.tree.leaves(state)):
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
    assert (generation / "manifest.json").is_file()


@pytest.mark.parametrize(
    "field,value",
    [
        ("config_digest", "different"),
        ("input_digest", "different"),
        ("dtype", "float32"),
        ("topology", {"processes": 2, "devices": 1}),
        ("runtime", {"jax": "different"}),
        ("source_digest", "different"),
    ],
)
def test_incompatible_restart_rejected(tmp_path, field, value):
    save_checkpoint(tmp_path, jnp.ones(2), metadata())
    expected = metadata() | {field: value}
    with pytest.raises(ValueError, match="incompatible"):
        load_checkpoint(tmp_path, expected)


def test_failure_before_latest_publication_keeps_previous_state(tmp_path, monkeypatch):
    import erot.runtime.checkpoint as checkpoint

    save_checkpoint(tmp_path, jnp.array([1.0]), metadata())
    original = checkpoint.os.replace

    def interrupted(source, target):
        if Path(target).name == "LATEST":
            raise OSError("simulated interruption")
        return original(source, target)

    monkeypatch.setattr(checkpoint.os, "replace", interrupted)
    with pytest.raises(OSError, match="simulated"):
        save_checkpoint(tmp_path, jnp.array([2.0]), metadata())
    recovered, _ = load_checkpoint(tmp_path, metadata())
    np.testing.assert_array_equal(recovered, [1.0])


def test_corrupted_payload_rejected(tmp_path):
    generation = save_checkpoint(tmp_path, jnp.ones(3), metadata())
    (generation / "arrays.npz").write_bytes(b"truncated")
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(tmp_path, metadata())


@pytest.mark.parametrize("mutation", ["schema", "type", "dtype"])
def test_manifest_schema_type_and_dtype_checked(tmp_path, mutation):
    generation = save_checkpoint(tmp_path, jnp.ones(3), metadata())
    path = generation / "manifest.json"
    manifest = json.loads(path.read_text())
    if mutation == "schema":
        manifest["schema_version"] = 900
    elif mutation == "type":
        manifest["tree"] = {"kind": "namedtuple", "name": "os.system", "items": []}
    else:
        manifest["arrays"]["a0"]["dtype"] = "int32"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path, metadata())


def test_latest_path_cannot_escape_checkpoint_directory(tmp_path):
    (tmp_path / "LATEST").write_text("../foreign")
    with pytest.raises(ValueError, match="generation"):
        load_checkpoint(tmp_path, metadata())


def test_missing_identity_and_object_arrays_rejected(tmp_path):
    with pytest.raises(ValueError, match="metadata"):
        save_checkpoint(tmp_path, jnp.ones(2), {})
    with pytest.raises(ValueError, match="numeric"):
        save_checkpoint(tmp_path, np.array([object()], dtype=object), metadata())
