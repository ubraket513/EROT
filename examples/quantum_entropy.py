"""Small noncommuting entropy-QOT example; run from an installed EROT environment."""

from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp

from erot.solvers.quantum_entropy import solve_quantum_entropy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=10000)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    a = jnp.array([[0.6, 0.1j], [-0.1j, 0.4]])
    b = jnp.array([[0.45, 0.07], [0.07, 0.55]])
    cost = jnp.array(
        [
            [0, 0.1j, 0.2, 0],
            [-0.1j, 0.3, 0, 0.1],
            [0.2, 0, 0.4, -0.1j],
            [0, 0.1, 0.1j, 0.2],
        ]
    )
    state, diagnostics = jax.jit(solve_quantum_entropy)(
        cost, a, b, args.epsilon, args.tolerance, args.max_iterations
    )
    state.coupling.block_until_ready()
    print(
        json.dumps(
            {
                "status": int(diagnostics.status),
                "iterations": int(diagnostics.iterations),
                "primal": float(diagnostics.primal),
                "dual": float(diagnostics.dual),
                "gap": float(diagnostics.gap),
                "marginal_max_error": float(diagnostics.feasibility),
                "minimum_eigenvalue": float(jnp.linalg.eigvalsh(state.coupling).min()),
            },
            indent=2,
        )
    )
    return 0 if int(diagnostics.status) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
