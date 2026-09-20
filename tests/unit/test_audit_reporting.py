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
