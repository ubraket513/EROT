"""Portable array selection without a scheduler or GPU."""

import json

import pytest

from erot.slurm import array_configuration


def test_array_selects_list_relative_path(tmp_path):
    folder = tmp_path / "configs"
    folder.mkdir()
    (folder / "trial.json").write_text('{"seed": 7}')
    manifest = tmp_path / "list.json"
    manifest.write_text(json.dumps(["configs/trial.json"]))
    assert array_configuration(manifest, "0") == {"seed": 7}


@pytest.mark.parametrize("index", ["-1", "1", "bad", None])
def test_array_rejects_invalid_index(tmp_path, index):
    manifest = tmp_path / "list.json"
    manifest.write_text('["trial.json"]')
    with pytest.raises(ValueError, match="array index"):
        array_configuration(manifest, index)


@pytest.mark.parametrize("content", [{}, [], [1]])
def test_array_requires_config_path_list(tmp_path, content):
    manifest = tmp_path / "list.json"
    manifest.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="configuration paths"):
        array_configuration(manifest, "0")
