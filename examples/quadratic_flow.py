"""Run a short quadratic-energy flow on a fixed three-cell grid."""

import jax
import jax.numpy as jnp

from erot.flows import Quadratic, initialize_flow, run_flow_chunk


def main():
    jax.config.update("jax_enable_x64", True)
    x = jnp.array([0.0, 0.5, 1.0])
    previous = jnp.array([0.2, 0.5, 0.3])
    energy = Quadratic(jnp.array([0.5, 0.2, 0.3]), weight=3.0)
    result = run_flow_chunk(
        initialize_flow(previous, backend="pdhg"),
        (x[:, None] - x[None, :]) ** 2,
        energy,
        0.4,
        1e-8,
        30000,
        steps=3,
        backend="pdhg",
        snapshot_stride=1,
    )
    jax.block_until_ready(result)
    print("Physical time:", float(result.state.time))
    print("Mass snapshots:", result.snapshots)
    print("Status:", int(result.state.status))
    return int(result.state.status != 0)


if __name__ == "__main__":
    raise SystemExit(main())
