"""The real benchmark must produce validated repeated-sample records."""

from __future__ import annotations

import pytest

from benchmarks.benchmark_solvers import benchmark_classical, benchmark_quantum
from benchmarks.reporting import assert_comparable, validate_record


@pytest.mark.parametrize(
    ("benchmark", "size"), [(benchmark_classical, 16), (benchmark_quantum, 2)]
)
def test_benchmark_records_include_accuracy_and_repeated_timings(benchmark, size):
    record = benchmark(size, "float64", "cpu")
    assert record["schema_version"] == 2
    assert len(record["samples_seconds"]) == 7
    assert record["first_call_seconds"] > 0
    assert record["first_call_seconds"] == record["compile_and_run_seconds"]
    assert record["case_digest"]
    assert record["output_policy"] == "dense-coupling"
    assert record["hardware_profile"]["devices"][0]["platform"] == "cpu"
    validate_record(record)
    assert_comparable(record, record)
