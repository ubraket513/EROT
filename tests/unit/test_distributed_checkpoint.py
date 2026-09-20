"""Global publication names exact durable rank generations."""

import json

import numpy as np
import pytest

from erot.runtime.distributed_checkpoint import (
    load_distributed_checkpoint,
    save_distributed_checkpoint,
)


def metadata():
    return {
        "config_digest": "c",
        "input_digest": "i",
        "dtype": "float64",
        "topology": {"processes": 1},
    }


def test_rank_roundtrip_and_exact_generation(tmp_path):
    state = {
        "f": np.array([1.0, 2.0]),
        "g": np.array([3.0]),
        "iterations": np.array(4, np.int32),
    }
    generation = save_distributed_checkpoint(tmp_path, state, metadata())
    manifest = json.loads((generation / "manifest.json").read_text())
    assert manifest["processes"] == 1
    assert len(manifest["ranks"]) == 1
    # Global manifest is authoritative, not a potentially changed rank LATEST.
    (generation / "rank-000000" / "LATEST").write_text("invalid\n")
    loaded, _ = load_distributed_checkpoint(tmp_path, metadata())
    for key in state:
        np.testing.assert_array_equal(loaded[key], state[key])


def test_interrupted_global_publication_keeps_prior(tmp_path, monkeypatch):
    import erot.runtime.distributed_checkpoint as module

    save_distributed_checkpoint(tmp_path, {"value": np.array(1)}, metadata())
    before = (tmp_path / "LATEST").read_text()
    original = module.os.replace

    def interrupt(source, destination):
        if destination == tmp_path / "LATEST":
            raise OSError("simulated publication interruption")
        return original(source, destination)

    monkeypatch.setattr(module.os, "replace", interrupt)
    with pytest.raises(RuntimeError, match="publication"):
        save_distributed_checkpoint(tmp_path, {"value": np.array(2)}, metadata())
    assert (tmp_path / "LATEST").read_text() == before
    loaded, _ = load_distributed_checkpoint(tmp_path, metadata())
    assert loaded["value"] == 1


def test_incompatible_and_corrupt_rank_rejected(tmp_path):
    generation = save_distributed_checkpoint(
        tmp_path, {"value": np.array(1)}, metadata()
    )
    with pytest.raises(RuntimeError, match="incompatible"):
        load_distributed_checkpoint(tmp_path, metadata() | {"input_digest": "changed"})
    manifest = json.loads((generation / "manifest.json").read_text())
    rank_manifest = (
        generation
        / "rank-000000"
        / manifest["ranks"][0]["generation"]
        / "manifest.json"
    )
    rank_manifest.write_text("{}")
    with pytest.raises(RuntimeError, match="checksum"):
        load_distributed_checkpoint(tmp_path, metadata())


@pytest.mark.parametrize(
    "change", [{"processes": 2}, {"schema_version": 99}, {"ranks": []}]
)
def test_global_manifest_topology_schema_and_completeness(tmp_path, change):
    generation = save_distributed_checkpoint(
        tmp_path, {"value": np.array(1)}, metadata()
    )
    path = generation / "manifest.json"
    manifest = json.loads(path.read_text())
    path.write_text(json.dumps(manifest | change))
    with pytest.raises(RuntimeError):
        load_distributed_checkpoint(tmp_path, metadata())
