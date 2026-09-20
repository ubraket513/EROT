"""Small dense equivalence and failure probes with explicit virtual CPU devices."""

import re

import jax
import jax.numpy as jnp
import numpy as np

from erot.geometry import PointCloudGeometry
from erot.geometry.plan import transport_objective
from erot.runtime.distributed import global_from_local, make_row_mesh
from erot.solvers.blocked_sinkhorn import solve_blocked_sinkhorn
from erot.solvers.distributed_sinkhorn import make_distributed_sinkhorn
from erot.solvers.sinkhorn import solve_sinkhorn
from erot.solvers.state import SinkhornState

jax.config.update("jax_enable_x64", True)
mesh = make_row_mesh()
assert mesh.size == 2
solve = make_distributed_sinkhorn(mesh, block_size=3)
rng = np.random.default_rng(8)
for dtype in (np.float32, np.float64):
    for concentrated, epsilon in ((False, 0.5), (True, 0.03)):
        x = np.pad(rng.normal(size=(7, 2)).astype(dtype), ((0, 1), (0, 0)))
        y = rng.normal(size=(5, 2)).astype(dtype)
        a = np.r_[rng.uniform(0.1, 1, 7), 0].astype(dtype)
        if concentrated:
            a[:4] = 0
        a /= a.sum()
        b = np.array([0.1, 0, 0.3, 0.2, 0.4], dtype=dtype)
        tolerance = 2e-5 if dtype == np.float32 else 1e-10
        arguments = (
            global_from_local(x, mesh),
            global_from_local(y, mesh, True),
            global_from_local(a, mesh),
            global_from_local(b, mesh, True),
            dtype(epsilon),
            dtype(tolerance),
        )
        initial = SinkhornState(
            (
                global_from_local(np.where(a > 0, 0, -np.inf).astype(dtype), mesh),
                global_from_local(
                    np.where(b > 0, 0, -np.inf).astype(dtype), mesh, True
                ),
            ),
            jnp.asarray(0, jnp.int32),
        )
        state, diagnostics = solve(*arguments, 20000, initial)
        jax.block_until_ready(state)
        assert int(diagnostics.status) == 0, diagnostics
        for shard in diagnostics.error.addressable_shards:
            np.testing.assert_array_equal(shard.data, np.asarray(diagnostics.error))
        host = jax.tree.map(lambda v: jnp.asarray(np.asarray(v)), state)
        geometry = PointCloudGeometry(jnp.asarray(x), jnp.asarray(y))
        dense = ((x[:, None] - y[None, :]) ** 2).sum(-1)
        reference, rd = solve_sinkhorn(
            jnp.asarray(dense),
            (jnp.asarray(a), jnp.asarray(b)),
            epsilon,
            tolerance,
            20000,
        )
        blocked, bd = solve_blocked_sinkhorn(
            geometry,
            (jnp.asarray(a), jnp.asarray(b)),
            epsilon,
            tolerance,
            20000,
            block_size=3,
        )
        assert int(rd.status) == int(bd.status) == 0
        atol = 2e-4 if dtype == np.float32 else 2e-8
        value = transport_objective(geometry, host.potentials, epsilon, block_size=3)
        for other in (reference, blocked):
            np.testing.assert_allclose(
                value,
                transport_objective(geometry, other.potentials, epsilon, block_size=3),
                atol=atol,
                rtol=atol,
            )
        f, g = map(np.asarray, host.potentials)
        assert abs(np.where(a > 0, f, 0) @ a) < atol
        plan = np.exp((f[:, None] + g[None, :] - dense) / epsilon)
        error = max(np.abs(plan.sum(1) - a).sum(), np.abs(plan.sum(0) - b).sum())
        np.testing.assert_allclose(error, diagnostics.error, atol=atol, rtol=atol)
        assert np.all(plan[-1] == 0)
        partial, pd = solve(*arguments, 3, initial)
        resumed, _ = solve(*arguments, 20000, partial)
        for actual, expected in zip(jax.tree.leaves(resumed), jax.tree.leaves(state)):
            np.testing.assert_allclose(actual, expected, atol=atol, rtol=atol)
        unchanged, _ = solve(*arguments, 0, partial)
        for actual, expected in zip(
            jax.tree.leaves(unchanged), jax.tree.leaves(partial)
        ):
            np.testing.assert_array_equal(actual, expected)
        bad = a.copy()
        bad[0] = -1
        failed, fd = solve(
            *arguments[:2], global_from_local(bad, mesh), *arguments[3:], 20, initial
        )
        assert int(fd.status) == 2 and int(failed.iterations) == 0
        _, fd = solve(*arguments, -1, initial)
        assert int(fd.status) == 2
try:
    solve(arguments[0].astype(jnp.complex128), *arguments[1:], 1, initial)
except ValueError as exc:
    assert "real floating" in str(exc)
else:
    raise AssertionError("complex classical points accepted")

# A finite-coordinate check on only one shard must become a global invalid status.
bad_x = x.copy()
bad_x[0, 0] = np.nan
_, invalid = solve(global_from_local(bad_x, mesh), *arguments[1:], 10, initial)
assert int(invalid.status) == 2
_, invalid = solve(
    *arguments, 1, initial._replace(iterations=jnp.asarray(2**31 - 1, jnp.int64))
)
assert int(invalid.status) == 2
for shard in invalid.status.addressable_shards:
    assert int(shard.data) == 2

# Compiler buffers contain row-sharded geometry and tiles, not full local plans.
memory_totals = []
for n, m in ((258, 383), (514, 766)):
    points = global_from_local(np.zeros((n, 2), np.float64), mesh)
    targets = global_from_local(np.zeros((m, 2), np.float64), mesh, True)
    a = global_from_local(np.full(n, 1 / n), mesh)
    b = global_from_local(np.full(m, 1 / m), mesh, True)
    initial = SinkhornState(
        (
            global_from_local(np.zeros(n), mesh),
            global_from_local(np.zeros(m), mesh, True),
        ),
        jnp.asarray(0, jnp.int32),
    )
    executable = solve.lower(points, targets, a, b, 0.2, 1e-8, 10, initial).compile()
    text = executable.as_text()
    shapes = {
        tuple(int(part) for part in shape.split(",") if part)
        for shape in re.findall(r"(?:bf|f|s|u|c)[0-9]+\[([0-9,]*)\]", text)
    }
    for rows in (n, n // 2):
        assert not any(
            shape == (rows * m,)
            or (len(shape) >= 2 and shape[:2] in ((rows, m), (m, rows)))
            for shape in shapes
        )
    memory = executable.memory_analysis()
    total = (
        memory.argument_size_in_bytes
        + memory.output_size_in_bytes
        + memory.temp_size_in_bytes
        - memory.alias_size_in_bytes
    )
    print("compiler bytes", n, m, total)
    assert total < (n // 2) * m * 8
    memory_totals.append(total)
assert memory_totals[1] < 2.6 * memory_totals[0]
print("collective solver probe passed")
