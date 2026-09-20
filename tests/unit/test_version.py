"""Version reporting must follow the installed distribution after a release."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import erot


def test_version_tracks_distribution_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = tmp_path / "erot-9.8.7.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.4\nName: EROT\nVersion: 9.8.7\n",
        encoding="utf-8",
    )
    try:
        with monkeypatch.context() as isolated:
            isolated.syspath_prepend(str(tmp_path))
            importlib.reload(erot)
            assert erot.__version__ == "9.8.7"
    finally:
        importlib.reload(erot)
