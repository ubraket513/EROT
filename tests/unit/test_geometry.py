"""Tiled squared-distance geometry and stable streamed reductions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp

from erot.geometry.dense import DenseGeometry
from erot.geometry.pointcloud import PointCloudGeometry
from erot.geometry.reductions import streamed_logsumexp

jax.config.update("jax_enable_x64", True)


def test_rectangular_tiles_and_padding_match_direct_differences():
    rng = np.random.default_rng(16)
    x, y = rng.normal(size=(5, 3)), rng.normal(size=(7, 3))
    expected = ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)
    for geom in (
        PointCloudGeometry(jnp.asarray(x), jnp.asarray(y)),
        DenseGeometry(jnp.asarray(expected)),
    ):
        actual = np.full((8, 8), np.inf)
        for i in (0, 4):
            for j in (0, 4):
                actual[i : i + 4, j : j + 4] = jax.jit(lambda g: g.cost_block(i, j, 4))(
                    geom
                )
        np.testing.assert_allclose(actual[:5, :7], expected, atol=1e-14)
        assert np.isinf(actual[5:, :]).all() and np.isinf(actual[:, 7:]).all()


def test_large_common_coordinate_offset_does_not_cancel_distances():
    x = jnp.array([[1e8, 1e8], [1e8 + 1, 1e8 + 2]])
    geom = PointCloudGeometry(x, x)
    np.testing.assert_array_equal(geom.cost_block(0, 0, 2), [[0.0, 5.0], [5.0, 0.0]])


@pytest.mark.parametrize("epsilon", [1.0, 1e-4])
def test_streaming_nondivisible_and_all_masked_tiles(epsilon):
    x, y = jnp.arange(5.0)[:, None], jnp.arange(7.0)[:, None]
    geom = PointCloudGeometry(x, y)
    f = jnp.array([0.1, -0.2, -jnp.inf, 0.3, 0.1])
    g = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, 0.2, -0.1, 0.4])
    expected = logsumexp(
        (
            np.asarray(f)[:, None]
            + np.asarray(g)[None, :]
            - (np.asarray(x) - np.asarray(y).T) ** 2
        )
        / epsilon,
        axis=1,
    )
    run = jax.jit(
        lambda geom, f, g: streamed_logsumexp(geom, f, g, epsilon, block_size=4)
    )
    np.testing.assert_allclose(run(geom, f, g), expected, atol=1e-10)
    reverse = streamed_logsumexp(geom.transpose(), g, f, epsilon, block_size=4)
    expected_reverse = logsumexp(
        (
            np.asarray(f)[:, None]
            + np.asarray(g)[None, :]
            - (np.asarray(x) - np.asarray(y).T) ** 2
        )
        / epsilon,
        axis=0,
    )
    np.testing.assert_allclose(reverse, expected_reverse, atol=1e-10)
    batched = jax.vmap(lambda left: run(geom, left, g))(jnp.stack([f, f + 0.1]))
    np.testing.assert_allclose(batched[0], expected, atol=1e-10)
    np.testing.assert_allclose(batched[1], expected + 0.1 / epsilon, atol=1e-10)


def test_geometry_shape_dtype_validation():
    with pytest.raises(ValueError):
        PointCloudGeometry(jnp.ones((2, 2)), jnp.ones((3, 1))).cost_block(0, 0, 2)
    with pytest.raises(ValueError):
        PointCloudGeometry(jnp.ones((0, 2)), jnp.ones((3, 2))).cost_block(0, 0, 2)
    with pytest.raises(ValueError):
        PointCloudGeometry(
            jnp.ones((2, 2), dtype=jnp.int32), jnp.ones((3, 2))
        ).cost_block(0, 0, 2)
    with pytest.raises(ValueError):
        DenseGeometry(jnp.ones((2, 2, 2))).cost_block(0, 0, 2)
    with pytest.raises(ValueError):
        DenseGeometry(jnp.ones((2, 3))).cost_block(0, 0, 0)
