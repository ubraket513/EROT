"""Public entropy-QOT invocation and compatibility checks."""

import json

import numpy as np
import pytest

from erot import SolverConfig, solve
from erot.cli import main


def inputs():
    return np.zeros((4, 4)), [np.diag([0.3, 0.7]), np.diag([0.6, 0.4])]


def test_host_entropy_quantum():
    c, marginals = inputs()
    result = solve(
        c,
        marginals,
        problem="quantum",
        regularizer="von_neumann",
        method="dual",
        config=SolverConfig(0.4, tolerance=1e-8),
    )
    assert result.converged
    np.testing.assert_allclose(result.coupling, np.kron(*marginals), atol=2e-8)


def test_invalid_entropy_state_does_not_claim_host_success():
    c, marginals = inputs()
    marginals[0] = np.diag([1.0, 0.0])
    result = solve(
        c,
        marginals,
        problem="quantum",
        regularizer="von_neumann",
        method="dual",
        config=SolverConfig(0.4),
    )
    assert not result.converged and np.isinf(result.error)


def test_quantum_cli_selects_entropy_and_preserves_default(tmp_path, capsys):
    c, marginals = inputs()
    np.save(tmp_path / "c.npy", c)
    np.save(tmp_path / "a.npy", marginals[0])
    np.save(tmp_path / "b.npy", marginals[1])
    args = [
        "solve",
        "quantum",
        "--cost",
        str(tmp_path / "c.npy"),
        "--marginal",
        str(tmp_path / "a.npy"),
        "--marginal",
        str(tmp_path / "b.npy"),
        "--epsilon",
        ".4",
        "--output",
        str(tmp_path / "out.npz"),
    ]
    assert main(args + ["--regularizer", "von_neumann", "--method", "dual"]) == 0
    assert json.loads(capsys.readouterr().out)["converged"]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["converged"]


@pytest.mark.parametrize(
    "regularizer,method", [("von_neumann", "cyclic"), ("quadratic", "dual")]
)
def test_unsupported_quantum_combinations(regularizer, method):
    c, marginals = inputs()
    with pytest.raises(ValueError, match="supported quantum combinations"):
        solve(
            c,
            marginals,
            problem="quantum",
            regularizer=regularizer,
            method=method,
            config=SolverConfig(0.4),
        )
