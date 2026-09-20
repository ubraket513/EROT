"""Fresh process isolation and independent experiment output ownership."""

import json
import os
import subprocess
import sys

import pytest

from erot.launcher import launch_experiments


def configs():
    return [
        {
            "name": "trial",
            "kind": "classical",
            "backend": "blocked",
            "n": 5,
            "m": 7,
            "seed": seed,
            "epsilon": 0.3,
            "tolerance": 1e-6,
            "chunk_size": 10,
            "max_iterations": 1000,
            "block_size": 3,
        }
        for seed in (1, 2)
    ]


def test_launcher_import_does_not_import_numerical_libraries():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import erot.launcher; assert 'jax' not in sys.modules; assert 'numpy' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("workers", [1, 2])
def test_independent_cpu_workers_and_affinity(tmp_path, workers):
    if len(os.sched_getaffinity(0)) < workers:
        pytest.skip("not enough allocated cores")
    report = launch_experiments(
        configs(),
        tmp_path,
        workers=workers,
        device="cpu",
        cpu_budget=workers,
        threads_per_worker=1,
    )
    assert report["completed"] == 2 and report["failed"] == 0
    assert len({r["run_id"] for r in report["runs"]}) == 2
    plans = {p["worker"]: p for p in report["resource_plan"]}
    for run in report["runs"]:
        saved = json.loads((tmp_path / run["run_id"] / "result.json").read_text())
        assert saved["status"] == "completed"
        assert saved["resources"]["affinity"] == plans[run["worker"]]["cpus"]
        assert saved["resources"]["thread_environment"]["OMP_NUM_THREADS"] == "1"


def test_duplicate_id_rejected_before_launch(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        launch_experiments(
            [configs()[0], configs()[0]], tmp_path, workers=1, device="cpu"
        )


def test_numerical_failure_preserves_worker_failure_exit(tmp_path):
    bad = configs()[0] | {"max_iterations": 1}
    report = launch_experiments([bad], tmp_path, workers=1, device="cpu", cpu_budget=1)
    assert report["failed"] == 1 and report["runs"][0]["returncode"] != 0
    assert (tmp_path / report["runs"][0]["run_id"] / "result.json").is_file()


def test_existing_run_requires_explicit_resume(tmp_path):
    settings = dict(workers=1, device="cpu", cpu_budget=1)
    first = launch_experiments(configs()[:1], tmp_path, **settings)
    assert first["completed"] == 1
    repeated = launch_experiments(configs()[:1], tmp_path, **settings)
    assert repeated["failed"] == 1 and repeated["completed"] == 0
    resumed = launch_experiments(configs()[:1], tmp_path, resume=True, **settings)
    assert resumed["completed"] == 1
    assert resumed["runs"][0]["run_id"] == first["runs"][0]["run_id"]


def test_array_template_cpu_execution(tmp_path):
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    config = tmp_path / "config.json"
    config.write_text(json.dumps(configs()[0]))
    manifest = tmp_path / "manifest.json"
    manifest.write_text('["config.json"]')
    env = dict(os.environ) | {
        "EROT_PYTHON": sys.executable,
        "EROT_CONFIG_LIST": str(manifest),
        "EROT_RESOURCE_PROFILE": str(root / "hpc/profiles/cpu.json"),
        "EROT_OUTPUT_ROOT": str(tmp_path / "results"),
        "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_CPUS_PER_TASK": "1",
    }
    completed = subprocess.run(
        ["bash", str(root / "hpc/run_array.sbatch")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["completed"] == 1
