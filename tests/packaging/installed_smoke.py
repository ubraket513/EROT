"""Run with an isolated interpreter outside the checkout, using base deps only."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

import numpy as np

import erot
import erot.experimental.classical
from erot.io import load_result


def run_cli(*args: str, cwd: Path) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, "-I", "-m", "erot", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return json.loads(result.stdout)


def main() -> None:
    package = Path(erot.__file__).resolve()
    assert package.is_relative_to(Path(sys.prefix).resolve()), package
    assert erot.__version__ == importlib.metadata.version("EROT")
    for optional in ("matplotlib", "cvxpy", "flax", "optax"):
        assert importlib.util.find_spec(optional) is None, optional

    console = Path(sysconfig.get_path("scripts")) / (
        "erot.exe" if os.name == "nt" else "erot"
    )
    help_result = subprocess.run(
        [str(console), "--help"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "solve" in help_result.stdout

    mass = np.array([0.25, 0.75])
    result = erot.solve(
        np.zeros((2, 2)),
        [mass, mass],
        problem="classical",
        regularizer="shannon",
        method="sinkhorn",
        config=erot.SolverConfig(epsilon=1.0, device="cpu"),
    )
    assert result.converged
    np.testing.assert_allclose(result.coupling, np.outer(mass, mass), atol=1e-8)

    with tempfile.TemporaryDirectory(prefix="erot-installed-workflow-") as directory:
        cwd = Path(directory)
        run_cli("generate", "cost", "--n", "3", "--output", "cost.npy", cwd=cwd)
        run_cli("generate", "marginal", "--n", "3", "--output", "mass.npy", cwd=cwd)
        record = run_cli(
            "solve",
            "classical",
            "--cost",
            "cost.npy",
            "--marginal",
            "mass.npy",
            "--marginal",
            "mass.npy",
            "--regularizer",
            "shannon",
            "--method",
            "sinkhorn",
            "--epsilon",
            "1",
            "--device",
            "cpu",
            "--output",
            "result.npz",
            cwd=cwd,
        )
        assert record["converged"]
        saved = load_result(cwd / "result.npz")
        assert saved["converged"]
        assert saved["metadata"]["problem"] == "classical"
        expected_mass = np.load(cwd / "mass.npy", allow_pickle=False)
        plan = saved["coupling"]
        assert np.isfinite(plan).all()
        assert (plan >= 0).all()
        np.testing.assert_allclose(plan.sum(axis=0), expected_mass, atol=1e-8)
        np.testing.assert_allclose(plan.sum(axis=1), expected_mass, atol=1e-8)
    print(
        f"Installed API and CLI workflow passed: EROT {erot.__version__} at {package}"
    )


if __name__ == "__main__":
    main()
