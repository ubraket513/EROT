"""Runtime metadata remains portable without optional command-line tools."""

import pytest

from erot.runtime.environment import source_identity


def test_source_identity_does_not_require_git(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("git unavailable")

    monkeypatch.setattr("erot.runtime.environment.subprocess.run", missing)
    identity = source_identity()
    assert identity["revision"] is None and len(identity["digest"]) == 64


def test_schema_version_is_not_boolean():
    from erot.runtime.config import normalize_config

    with pytest.raises(ValueError):
        normalize_config({"schema_version": True})
