"""Inspect built release artifacts, not the source-tree import path."""

from __future__ import annotations

import os
import tarfile
import zipfile
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


def test_wheel_contains_only_package_and_distribution_metadata(
    artifacts: tuple[Path, Path],
) -> None:
    with zipfile.ZipFile(artifacts[0]) as wheel:
        names = set(wheel.namelist())
    assert {
        "erot/__init__.py",
        "erot/__main__.py",
        "erot/api.py",
        "erot/classical.py",
        "erot/quantum.py",
        "erot/cli.py",
        "erot/experimental/__init__.py",
        "erot/experimental/classical.py",
    } <= names
    for name in names:
        first = PurePosixPath(name).parts[0]
        assert first == "erot" or first.endswith(".dist-info"), name
    assert any(name.endswith(".dist-info/licenses/LICENSE.txt") for name in names)
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
        "LICENSE.txt",
        "src/erot/__init__.py",
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
