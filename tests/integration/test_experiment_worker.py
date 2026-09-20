"""Deterministic worker state and scientific restart behavior."""

import json

import jax
import numpy as np
import pytest

from erot.experiments import normalize_config, run_experiment
from erot.runtime.checkpoint import load_checkpoint

jax.config.update("jax_enable_x64", True)


def config(**changes):
    return {
        "schema_version": 1,
        "name": "test",
        "kind": "flow",
        "backend": "blocked",
        "energy": "quadratic",
        "n": 5,
        "seed": 9,
        "epsilon": 0.2,
        "time_step": 0.3,
        "tolerance": 1e-6,
        "max_iterations": 1000,
        "steps": 3,
        "chunk_size": 1,
        "block_size": 3,
        "snapshot_stride": 1,
        **changes,
    }


def read_state(path):
    record = json.loads((path / "run.json").read_text())
    return load_checkpoint(path / "checkpoints", record["checkpoint_metadata"])[0]


@pytest.mark.parametrize("kind", ["flow", "classical", "quantum"])
def test_worker_resume_matches_uninterrupted(tmp_path, kind):
    c = config(kind=kind)
    if kind == "classical":
        c.update(backend="blocked", m=7, chunk_size=2, max_iterations=1000)
    elif kind == "quantum":
        c.update(backend="entropy", n=2, m=2, chunk_size=2, max_iterations=1000)
    whole = run_experiment(c, tmp_path / "whole")
    partial = run_experiment(c, tmp_path / "resume", stop_after_chunks=1)
    assert partial["status"] == "checkpointed"
    resumed = run_experiment(c, tmp_path / "resume", resume=True)
    assert whole["status"] == resumed["status"] == "completed"
    a, b = read_state(tmp_path / "whole"), read_state(tmp_path / "resume")
    for left, right in zip(jax.tree.leaves(a["state"]), jax.tree.leaves(b["state"])):
        np.testing.assert_allclose(left, right, rtol=0, atol=1e-12)
    assert a["chunks"] == b["chunks"]


def test_configuration_change_and_output_reuse_rejected(tmp_path):
    c = config()
    run_experiment(c, tmp_path, stop_after_chunks=1)
    with pytest.raises(ValueError, match="incompatible"):
        run_experiment(c | {"epsilon": 0.3}, tmp_path, resume=True)
    with pytest.raises(ValueError, match="resume"):
        run_experiment(c, tmp_path)


def test_failure_is_checkpointed_without_physical_time_advance(tmp_path):
    c = config(inner_iterations=0)
    result = run_experiment(c, tmp_path)
    assert result["status"] == "failed"
    saved = read_state(tmp_path)["state"]
    assert saved.time == 0 and saved.accepted_steps == 0
    assert saved.status != 0


def test_run_lock_prevents_duplicate_writer(tmp_path):
    from erot.runtime.environment import run_lock

    with run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="owned"):
            run_experiment(config(), tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 99},
        {"n": 0},
        {"dtype": "float16"},
        {"epsilon": float("nan")},
        {"chunk_size": 1.5},
        {"name": "../escape"},
        {"unexpected": True},
    ],
)
def test_configuration_rejects_invalid_controls(change):
    with pytest.raises(ValueError):
        normalize_config(config(**change))


@pytest.mark.parametrize(
    "kind,backend,metric",
    [
        ("flow", "pdhg", "energy"),
        ("flow", "sinkhorn", "energy"),
        ("classical", "sinkhorn", "objective"),
        ("quantum", "quadratic", "objective"),
    ],
)
def test_supported_workloads_report_scientific_metrics(tmp_path, kind, backend, metric):
    c = config(
        kind=kind,
        backend=backend,
        steps=1,
        max_iterations=20000,
        chunk_size=1 if kind == "flow" else 500,
    )
    if kind == "quantum":
        c.update(n=2, m=2)
    report = run_experiment(c, tmp_path)
    assert report["status"] == "completed"
    assert np.isfinite(report["scientific"][metric])


def test_flow_timing_tracks_static_count_and_state_specializations(tmp_path):
    report = run_experiment(
        config(backend="pdhg", steps=7, chunk_size=2, max_iterations=20000), tmp_path
    )
    assert report["status"] == "completed"
    events = [
        json.loads(line)
        for line in (tmp_path / "timing.jsonl").read_text().splitlines()
    ]
    assert len(events) == 4
    # Middle chunks repeat the same signature; final count=1 is a new variant.
    assert ["compile_and_execute_seconds" in event for event in events] == [
        True,
        False,
        False,
        True,
    ]
