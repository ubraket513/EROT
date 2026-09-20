"""Energy/proximal conventions are checked against independent scalar roots."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq

from erot.flows.functionals import Entropy, Potential, Quadratic
from erot.optim.prox import entropy_prox

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("alpha", [0.0, 1e-8, 1e-4, 0.1, 1.0, 1e4])
def test_entropy_prox_matches_independent_log_root(alpha):
    z = np.array([-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 1e4])
    volumes = np.geomspace(0.1, 10.0, len(z))
    actual = np.asarray(
        jax.jit(entropy_prox)(jnp.asarray(z), alpha, jnp.asarray(volumes))
    )
    if alpha == 0:
        expected = np.maximum(z, 0.0)
    else:
        expected = []
        for value, w in zip(z, volumes, strict=True):
            lower = min(value / alpha + np.log(w) - 1, np.log(alpha) - 1, -1.0)
            upper = max(np.log(max(value, 1.0)), np.log(w), 0.0) + 1
            root = brentq(
                lambda t: np.exp(t) + alpha * (t - np.log(w)) - value, lower, upper
            )
            expected.append(np.exp(root))
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=1e-12)


def test_entropy_cell_volume_value_and_gradient():
    rho = jnp.array([0.2, 0.0, 0.8])
    w = jnp.array([0.1, 0.4, 0.5])
    energy = Entropy(w)
    expected = 0.2 * (np.log(2) - 1) + 0.8 * (np.log(1.6) - 1)
    assert energy.value(rho) == pytest.approx(expected)
    positive = jnp.array([0.2, 0.3, 0.5])
    np.testing.assert_allclose(
        jax.grad(energy.value)(positive), energy.gradient(positive), atol=1e-12
    )
    assert np.isinf(energy.value(jnp.array([-0.1, 0.3, 0.8])))


def test_quadratic_and_potential_proximal_stationarity():
    z = jnp.array([-0.5, 0.4, 1.2])
    target = jnp.array([0.2, 0.3, 0.5])
    alpha = 0.3
    for energy in (Quadratic(target, 2.0), Potential(jnp.array([0.1, 0.2, 0.3]))):
        x = energy.prox(z, alpha)
        residual = x - z + alpha * energy.gradient(x)
        assert jnp.all(x >= 0)
        assert (
            jnp.max(jnp.abs(jnp.where(x > 0, residual, jnp.minimum(residual, 0))))
            < 1e-12
        )
        np.testing.assert_allclose(
            jax.grad(energy.value)(target), energy.gradient(target), atol=1e-12
        )


def test_energy_objects_are_transformable_pytrees():
    energy = Entropy(jnp.array([0.2, 0.8]))
    got = jax.jit(lambda e, r: e.value(r))(energy, jnp.array([0.4, 0.6]))
    assert np.isfinite(got)


@pytest.mark.parametrize("mass,volume", [(1e100, 1e-300), (1e-100, 1e300)])
def test_entropy_extreme_log_ratios_stay_finite(mass, volume):
    energy = Entropy(jnp.array([volume]))
    rho = jnp.array([mass])
    expected = np.log(mass) - np.log(volume)
    np.testing.assert_allclose(energy.gradient(rho), [expected], rtol=1e-14)
    assert float(energy.value(rho)) == pytest.approx(
        mass * (expected - 1), rel=1e-14, abs=0.0
    )


def test_zero_weight_entropy_has_zero_autodiff_at_zero_mass():
    energy = Entropy(jnp.array([0.1, 0.2]), 0.0)
    np.testing.assert_array_equal(
        jax.grad(energy.value)(jnp.array([0.0, 1.0])), [0.0, 0.0]
    )
