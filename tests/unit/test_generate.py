from __future__ import annotations

import numpy as np
import pytest

from erot.generate import cost_tensor, gaussian_marginal


def test_generators_are_finite_and_normalized() -> None:
    for kind in ("euclidean", "weak-coulomb", "strong-coulomb"):
        cost = cost_tensor(4, 3, kind=kind)
        assert cost.shape == (4, 4, 4)
        assert np.all(np.isfinite(cost))
    marginal = gaussian_marginal(100, locs=(-1.0, 1.0), scales=(0.5, 0.75))
    assert marginal.shape == (100,)
    assert marginal.sum() == pytest.approx(1.0)
    assert np.all(marginal >= 0)


def test_generator_validation() -> None:
    with pytest.raises(ValueError):
        cost_tensor(0)
    with pytest.raises(ValueError):
        gaussian_marginal(10, locs=(0.0,), scales=(-1.0,))
