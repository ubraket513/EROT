"""Fresh-process runtime smoke; invoked by the integration test."""

import sys

from erot.runtime.distributed import (
    global_from_local,
    initialize_runtime,
    make_row_mesh,
)

assert "jax" not in sys.modules
initialize_runtime()

import jax  # noqa: E402
import numpy as np  # noqa: E402

mesh = make_row_mesh()
assert mesh.size == 2
x = global_from_local(np.arange(12, dtype=np.float32).reshape(6, 2), mesh)
y = global_from_local(np.arange(5, dtype=np.float32), mesh, replicated=True)
assert x.shape == (6, 2) and y.shape == (5,)
assert [shard.data.shape for shard in x.addressable_shards] == [(3, 2), (3, 2)]
for shard in y.addressable_shards:
    np.testing.assert_array_equal(shard.data, np.arange(5, dtype=np.float32))
np.testing.assert_array_equal(
    np.asarray(x), np.arange(12, dtype=np.float32).reshape(6, 2)
)
try:
    initialize_runtime("localhost:12345", 2, 0)
except RuntimeError as exc:
    assert "initialize" in str(exc).lower()
else:
    raise AssertionError("late distributed initialization was accepted")
print("runtime probe passed", jax.__version__)
