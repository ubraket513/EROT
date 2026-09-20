from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from erot.cli import main
from erot.io import load_result


def test_generate_solve_and_plot_cli(tmp_path: Path, capsys: object) -> None:
    cost_path = tmp_path / "cost.npy"
    marginal_path = tmp_path / "marginal.npy"
    result_path = tmp_path / "result.npz"
    plot_path = tmp_path / "coupling.png"

    assert main(["generate", "cost", "--n", "3", "--output", str(cost_path)]) == 0
    assert (
        main(["generate", "marginal", "--n", "3", "--output", str(marginal_path)]) == 0
    )
    assert (
        main(
            [
                "solve",
                "classical",
                "--cost",
                str(cost_path),
                "--marginal",
                str(marginal_path),
                "--marginal",
                str(marginal_path),
                "--regularizer",
                "shannon",
                "--method",
                "sinkhorn",
                "--epsilon",
                "1",
                "--device",
                "cpu",
                "--output",
                str(result_path),
            ]
        )
        == 0
    )
    assert main(["plot", str(result_path), "--output", str(plot_path)]) == 0

    assert cost_path.exists()
    assert marginal_path.exists()
    assert result_path.exists()
    assert plot_path.exists()
    archive = load_result(result_path)
    assert archive["converged"]
    assert archive["metadata"]["problem"] == "classical"

    lines = capsys.readouterr().out.strip().splitlines()  # type: ignore[attr-defined]
    assert all("output" in json.loads(line) for line in lines)


def test_result_archive_disallows_pickle(tmp_path: Path) -> None:
    invalid = tmp_path / "bad.npz"
    np.savez(invalid, coupling=np.zeros((2, 2)))
    try:
        load_result(invalid)
    except ValueError as exc:
        assert "missing" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid result archive was accepted")
