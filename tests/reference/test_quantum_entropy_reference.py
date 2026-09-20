"""Independent NumPy/SciPy and classical reductions for quantum entropy."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from erot.solvers.quantum_entropy import solve_quantum_entropy

minimize = pytest.importorskip("scipy.optimize").minimize
logsumexp = pytest.importorskip("scipy.special").logsumexp

jax.config.update("jax_enable_x64", True)
pytestmark = pytest.mark.reference


def test_diagonal_classical_reduction():
    cost = np.array([[0.0, 0.4, 0.8], [0.7, 0.2, 0.1]])
    a, b, eps = np.array([0.4, 0.6]), np.array([0.2, 0.3, 0.5]), 0.3
    kernel = np.exp(-cost / eps)
    u, v = np.ones(2), np.ones(3)
    for _ in range(1000):
        u = a / (kernel @ v)
        v = b / (kernel.T @ u)
    expected = u[:, None] * kernel * v[None, :]
    s, d = solve_quantum_entropy(
        jnp.diag(jnp.asarray(cost.ravel())),
        jnp.diag(jnp.asarray(a)),
        jnp.diag(jnp.asarray(b)),
        eps,
        1e-9,
        10000,
    )
    assert d.status == 0
    np.testing.assert_allclose(s.coupling, np.diag(expected.ravel()), atol=3e-9)
    objective = np.sum(cost * expected) + eps * np.sum(
        expected * (np.log(expected) - 1)
    )
    np.testing.assert_allclose(d.primal, objective, atol=3e-9)


def test_complex_noncommuting_against_independent_scipy_dual():
    rng = np.random.default_rng(212)
    raw = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    cost = (raw + raw.conj().T) * 0.1
    a = np.array([[0.6, 0.06 + 0.09j], [0.06 - 0.09j, 0.4]])
    b = np.array([[0.45, -0.07j], [0.07j, 0.55]])
    eps = 0.5
    pauli = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    basis = np.concatenate(
        [[np.kron(p, np.eye(2)) for p in pauli], [np.kron(np.eye(2), p) for p in pauli]]
    )
    targets = np.r_[
        [np.trace(a @ p).real for p in pauli], [np.trace(b @ p).real for p in pauli]
    ]

    def evaluate(x):
        h = np.einsum("k,kij->ij", x, basis) - cost
        w, q = np.linalg.eigh(h / eps)
        z = logsumexp(w)
        rho = (q * np.exp(w - z)) @ q.conj().T
        dual = targets @ x - eps * (z + 1)
        gradient = targets - np.einsum("kij,ji->k", basis, rho).real
        return -dual, -gradient, rho

    ref = minimize(
        lambda x: evaluate(x)[:2],
        np.zeros(6),
        jac=True,
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 2000},
    )
    assert np.max(np.abs(evaluate(ref.x)[1])) < 2e-8
    state, diag = solve_quantum_entropy(
        jnp.asarray(cost), jnp.asarray(a), jnp.asarray(b), eps, 1e-9, 10000
    )
    assert diag.status == 0
    np.testing.assert_allclose(state.coupling, evaluate(ref.x)[2], atol=2e-8)
    np.testing.assert_allclose(diag.dual, -ref.fun, atol=2e-9)
    w = np.linalg.eigvalsh(np.asarray(state.coupling))
    primal = np.trace(cost @ state.coupling).real + eps * np.sum(w * (np.log(w) - 1))
    np.testing.assert_allclose(diag.primal, primal, atol=1e-12)
