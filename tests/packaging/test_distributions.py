"""Inspect built release artifacts, not the source-tree import path."""

from __future__ import annotations

import configparser
import os
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath

import pytest

pytestmark = pytest.mark.packaging


@pytest.fixture(scope="module")
def artifacts() -> tuple[Path, Path]:
    configured = os.environ.get("EROT_DIST_DIR")
    if configured is None:
        pytest.skip("set EROT_DIST_DIR to inspect built wheel and sdist artifacts")
    directory = Path(configured).resolve()
    wheels = list(directory.glob("*.whl"))
    sources = list(directory.glob("*.tar.gz"))
    assert len(wheels) == len(sources) == 1, (wheels, sources)
    return wheels[0], sources[0]


@pytest.fixture(params=["rebuilt", "direct"])
def wheel_path(request, artifacts):
    if request.param == "rebuilt":
        return artifacts[0]
    configured = os.environ.get("EROT_DIRECT_DIST_DIR")
    if configured is None:
        pytest.skip("set EROT_DIRECT_DIST_DIR to inspect the direct wheel")
    wheels = list(Path(configured).resolve().glob("*.whl"))
    assert len(wheels) == 1, wheels
    return wheels[0]


def test_wheel_contains_only_package_and_distribution_metadata(wheel_path) -> None:
    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
        metadata = Parser().parsestr(
            wheel.read(
                next(name for name in names if name.endswith(".dist-info/METADATA"))
            ).decode()
        )
        scripts = configparser.ConfigParser()
        scripts.read_string(
            wheel.read(
                next(
                    name
                    for name in names
                    if name.endswith(".dist-info/entry_points.txt")
                )
            ).decode()
        )
    assert metadata["License-Expression"] == "MIT AND Apache-2.0"
    assert metadata["Requires-Python"] == ">=3.11"
    assert dict(scripts["console_scripts"]) == {
        "erot": "erot.cli:main",
        "erot-run": "erot.experiments:main",
        "erot-launch": "erot.launcher:main",
        "erot-array": "erot.slurm:main",
        "erot-distributed": "erot.distributed:main",
    }
    assert {
        "erot/__init__.py",
        "erot/__main__.py",
        "erot/api.py",
        "erot/classical.py",
        "erot/quantum.py",
        "erot/cli.py",
        "erot/flows/__init__.py",
        "erot/geometry/pointcloud.py",
        "erot/operators/quantum.py",
        "erot/solvers/quantum_entropy.py",
        "erot/solvers/distributed_sinkhorn.py",
        "erot/runtime/checkpoint.py",
        "erot/runtime/distributed_checkpoint.py",
        "erot/experiments.py",
        "erot/launcher.py",
        "erot/slurm.py",
        "erot/distributed.py",
        "erot/experimental/__init__.py",
        "erot/experimental/classical.py",
    } <= names
    for name in names:
        first = PurePosixPath(name).parts[0]
        assert first == "erot" or first.endswith(".dist-info"), name
    assert any(name.endswith(".dist-info/licenses/LICENSE.txt") for name in names)
    assert any(
        name.endswith(".dist-info/licenses/LICENSES/Apache-2.0.txt") for name in names
    )
    assert any(name.endswith(".dist-info/licenses/NOTICE") for name in names)
    assert any(name.endswith(".dist-info/entry_points.txt") for name in names)
    assert not any(name.endswith((".pt", ".png", ".ipynb", ".pyc")) for name in names)


def test_sdist_excludes_local_repositories_and_generated_data(
    artifacts: tuple[Path, Path],
) -> None:
    with tarfile.open(artifacts[1]) as source:
        names = {
            str(
                PurePosixPath(member.name).relative_to(
                    PurePosixPath(member.name).parts[0]
                )
            )
            for member in source.getmembers()
        }
    assert {
        "pyproject.toml",
        "CHANGELOG.md",
        "docs/migration-0.2.md",
        "docs/support-matrix.md",
        "requirements/base-minimum.txt",
        "LICENSE.txt",
        "LICENSES/Apache-2.0.txt",
        "NOTICE",
        "src/erot/__init__.py",
        "examples/heat_flow.py",
        "examples/quadratic_flow.py",
        "examples/quantum_entropy.py",
        "hpc/run_array.sbatch",
        "hpc/run_distributed.sbatch",
        "hpc/distributed_rank.sh",
        "experiments/configs/distributed-classical.json",
        "requirements/distributed-cpu.txt",
        "docs/performance/native-decision.md",
        "src/erot/experimental/classical.py",
        "tests/unit/test_api.py",
        "tests/packaging/installed_smoke.py",
    } <= names
    forbidden = {
        "archive",
        "QOTLib",
        "numerical-gradient-flows",
        "SDP-Simulation",
        ".git",
        ".serena",
        ".superpowers",
        ".worktrees",
        "benchmark-results",
        "results",
        "checkpoints",
        "__pycache__",
    }
    for name in names:
        parts = PurePosixPath(name).parts
        assert not forbidden.intersection(parts), name
        assert not any(part.startswith(".venv") for part in parts), name
        assert not name.endswith((".pt", ".png", ".ipynb", ".pyc")), name
