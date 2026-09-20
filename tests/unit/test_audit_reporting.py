"""An audit must retain nonfinite results without emitting invalid JSON."""

import json

from benchmarks.audit_legacy_flow import run_check


def test_nonfinite_audit_result_is_retained_as_a_failure():
    record = run_check(
        "unstable",
        lambda: {
            "status": "passed",
            "objective": float("nan"),
            "values": [1.0, float("inf")],
        },
    )
    assert record["status"] == "mismatch"
    assert record["objective"] is None
    assert record["values"] == [1.0, None]
    json.dumps(record, allow_nan=False)


def test_pdhg_audit_rejects_signed_plan_with_correct_marginals_and_objective():
    from types import SimpleNamespace

    import numpy as np

    from benchmarks.audit_legacy_flow import audit_pdhg

    previous = np.array([0.2, 0.5, 0.3])
    signed = np.diag(previous) + np.outer([1, -2, 1], [1, -1, 0])
    legacy = SimpleNamespace(
        proxF_quadratic=lambda z, alpha, **kwargs: z,
        pdhg_jko=lambda *args, **kwargs: SimpleNamespace(rho=previous, pi=signed),
    )
    report = audit_pdhg(legacy, "functional")
    assert report["row_residual"] < 1e-12
    assert report["column_residual"] < 1e-12
    assert report["status"] == "mismatch"


def test_block_audit_rejects_indefinite_slack_even_with_small_objective_gap():
    import numpy as np

    from benchmarks.audit_qotlib import slack_feasibility

    cost = np.diag([0.0, 1.0, 1.0, 0.0])
    lifted_dual = np.kron(np.diag([0.01, -0.01]), np.eye(2))
    slack = cost - lifted_dual
    diagnostics = slack_feasibility(slack[None, :, :], 0.0)
    assert diagnostics["slack_min_eigenvalue"] == -0.01
    assert not diagnostics["feasible"]
