"""Two-process CPU collective equivalence, each rank supplies only local rows."""

import json
import sys
from pathlib import Path

from erot.runtime.distributed import initialize_runtime

initialize_runtime(sys.argv[1], 2, int(sys.argv[2]), [0])

import jax  # noqa: E402
import numpy as np  # noqa: E402

from erot.runtime.distributed import global_from_local, make_row_mesh  # noqa: E402
from erot.solvers.distributed_sinkhorn import make_distributed_sinkhorn  # noqa: E402
from erot.solvers.state import SinkhornState  # noqa: E402

jax.config.update("jax_enable_x64", True)
rank = jax.process_index()
mesh = make_row_mesh()
assert jax.device_count() == 2 and jax.local_device_count() == 1
# Seven valid source rows, four per rank, and a replicated target cloud.
indices = np.arange(rank * 4, (rank + 1) * 4)
x = global_from_local((indices / 7)[:, None], mesh)
y = global_from_local(np.linspace(0, 1, 5)[:, None], mesh, True)
a_local = np.where(indices < 7, 1 / 7, 0)
a = global_from_local(a_local, mesh)
b = global_from_local(np.full(5, 0.2), mesh, True)
state = SinkhornState(
    (
        global_from_local(np.where(a_local > 0, 0.0, -np.inf), mesh),
        global_from_local(np.zeros(5), mesh, True),
    ),
    global_from_local(np.asarray(0, np.int32), mesh, True),
)
solve = make_distributed_sinkhorn(mesh, block_size=3)
initial = state
state, diagnostics = solve(x, y, a, b, 0.2, 1e-10, 2000, initial)
jax.block_until_ready(state)
import erot.runtime.distributed_checkpoint as checkpoint_module  # noqa: E402
from erot.runtime.distributed_checkpoint import (  # noqa: E402
    collective_call,
    load_distributed_checkpoint,
    save_distributed_checkpoint,
)

root = Path(sys.argv[3]).parent / "checkpoints"
metadata = {
    "config_digest": "probe",
    "input_digest": "rows7targets5",
    "dtype": "float64",
    "topology": {"processes": 2, "devices": 2},
}
partial, _ = solve(x, y, a, b, 0.2, 1e-10, 3, initial)
local = jax.tree.map(
    lambda value: np.asarray(value.addressable_shards[0].data), partial
)
save_distributed_checkpoint(root, local, metadata)
loaded, _ = load_distributed_checkpoint(root, metadata)
restored = SinkhornState(
    (
        global_from_local(loaded.potentials[0], mesh),
        global_from_local(loaded.potentials[1], mesh, True),
    ),
    global_from_local(loaded.iterations, mesh, True),
)
resumed, _ = solve(x, y, a, b, 0.2, 1e-10, 2000, restored)
for actual, expected in zip(jax.tree.leaves(resumed), jax.tree.leaves(state)):
    np.testing.assert_allclose(
        actual.addressable_shards[0].data,
        expected.addressable_shards[0].data,
        atol=1e-12,
        rtol=0,
    )
# One rank's publication failure is agreed globally; the old pointer survives.
before = (root / "LATEST").read_text()
original = checkpoint_module.save_checkpoint
if rank == 1:

    def fail(*args, **kwargs):
        raise OSError("simulated rank write interruption")

    checkpoint_module.save_checkpoint = fail
try:
    save_distributed_checkpoint(root, local, metadata)
except RuntimeError as exc:
    assert "rank save" in str(exc)
else:
    raise AssertionError("incomplete rank checkpoint published")
finally:
    checkpoint_module.save_checkpoint = original
assert (root / "LATEST").read_text() == before
load_distributed_checkpoint(root, metadata)


# Missing data on rank1 must make every rank fail load, not proceed to a solve.
def corrupt_rank():
    if rank == 0:
        generation = root / before.strip()
        manifest = json.loads((generation / "manifest.json").read_text())
        payload = (
            generation
            / "rank-000001"
            / manifest["ranks"][1]["generation"]
            / "arrays.npz"
        )
        payload.unlink()


collective_call("test corruption", corrupt_rank)
try:
    load_distributed_checkpoint(root, metadata)
except RuntimeError as exc:
    assert "checkpoint load" in str(exc)
else:
    raise AssertionError("missing rank payload accepted")
# Access addressable data only; no global np.asarray of sharded source arrays.
record = {
    "rank": rank,
    "f": np.asarray(state.potentials[0].addressable_shards[0].data).tolist(),
    "g": np.asarray(state.potentials[1].addressable_shards[0].data).tolist(),
    "iterations": int(state.iterations.addressable_shards[0].data),
    "error": float(diagnostics.error.addressable_shards[0].data),
    "status": int(diagnostics.status.addressable_shards[0].data),
}
Path(sys.argv[3]).write_text(json.dumps(record))
jax.distributed.shutdown()
