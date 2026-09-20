"""Compiled storage evidence for the point-cloud path, distinct from GPU peaks."""

from functools import partial

import jax
import jax.numpy as jnp

from benchmarks.benchmark_blocked_sinkhorn import compiled_memory
from erot.geometry import PointCloudGeometry
from erot.solvers.blocked_sinkhorn import solve_blocked_sinkhorn

jax.config.update("jax_enable_x64", True)


def test_fixed_tile_compiled_storage_grows_linearly_not_quadratically():
    reports = []
    for n, m in ((257, 383), (514, 766)):
        geometry = PointCloudGeometry(jnp.zeros((n, 3)), jnp.ones((m, 3)))
        marginals = (jnp.ones(n) / n, jnp.ones(m) / m)
        function = jax.jit(partial(solve_blocked_sinkhorn, block_size=32))
        compiled = function.lower(geometry, marginals, 0.3, 1e-8, 1000).compile()
        report = compiled_memory(compiled, n, m)
        assert report["full_matrix_shapes"] == []
        assert report["compiled_total_bytes"] is not None
        assert report["compiled_total_bytes"] < n * m * 8
        reports.append(report)
    assert reports[1]["compiled_total_bytes"] < 2.6 * reports[0]["compiled_total_bytes"]


def test_failed_benchmark_preserves_strict_json_evidence():
    import json

    from benchmarks.benchmark_blocked_sinkhorn import run

    report = run(
        n=2, m=3, features=1, block_size=2, epsilon=1e-320, repeats=1, max_iterations=2
    )
    assert report["blocked"]["status"] != 0
    json.dumps(report, allow_nan=False)
