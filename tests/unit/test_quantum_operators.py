"""Independent index algebra for bipartite quantum operators."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from erot.operators.quantum import gibbs_state, partial_trace_adjoint, partial_traces

jax.config.update("jax_enable_x64", True)


def test_complex_unequal_partial_traces_and_adjoint():
    rng = np.random.default_rng(72)
    x = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    u = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    v = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    expected_a = np.array(
        [
            [sum(x[3 * i + k, 3 * j + k] for k in range(3)) for j in range(2)]
            for i in range(2)
        ]
    )
    expected_b = np.array(
        [
            [sum(x[3 * k + i, 3 * k + j] for k in range(2)) for j in range(3)]
            for i in range(3)
        ]
    )
    a, b = jax.jit(lambda x: partial_traces(x, (2, 3)))(jnp.asarray(x))
    np.testing.assert_allclose(a, expected_a, atol=1e-14)
    np.testing.assert_allclose(b, expected_b, atol=1e-14)
    lifted = partial_trace_adjoint((jnp.asarray(u), jnp.asarray(v)))
    np.testing.assert_allclose(lifted, np.kron(u, np.eye(3)) + np.kron(np.eye(2), v))
    np.testing.assert_allclose(np.vdot(x, lifted), np.vdot(a, u) + np.vdot(b, v))


def test_rank_one_product_traces():
    a = np.array([1, 1j]) / np.sqrt(2)
    b = np.array([1, 0, -1j]) / np.sqrt(2)
    rho_a, rho_b = np.outer(a, a.conj()), np.outer(b, b.conj())
    actual = partial_traces(jnp.asarray(np.kron(rho_a, rho_b)), (2, 3))
    np.testing.assert_allclose(actual[0], rho_a, atol=1e-14)
    np.testing.assert_allclose(actual[1], rho_b, atol=1e-14)


def test_gibbs_shift_and_degenerate_spectrum():
    h = jnp.array([[2.0, 1j], [-1j, -1.0]])
    rho, logz = gibbs_state(h, 0.3)
    shifted, shifted_logz = gibbs_state(h + 1e5 * jnp.eye(2), 0.3)
    np.testing.assert_allclose(shifted, rho, atol=1e-11)
    np.testing.assert_allclose(shifted_logz - logz, 1e5 / 0.3)
    np.testing.assert_allclose(np.trace(rho), 1, atol=1e-14)
    assert np.linalg.eigvalsh(rho).min() >= -1e-14
    uniform, logz = gibbs_state(jnp.zeros((6, 6)), 0.7)
    np.testing.assert_allclose(uniform, np.eye(6) / 6, atol=1e-14)
    np.testing.assert_allclose(logz, np.log(6))


def test_operator_shapes_are_explicit():
    with pytest.raises(ValueError):
        partial_traces(jnp.eye(6), (2, 2))
    with pytest.raises(ValueError):
        partial_traces(jnp.eye(6), (0, 3))
    with pytest.raises(ValueError):
        partial_trace_adjoint((jnp.ones((2, 3)), jnp.eye(3)))
    with pytest.raises(ValueError):
        gibbs_state(jnp.ones((2, 3)), 1.0)
