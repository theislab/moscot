import ot
import pytest

from jax import numpy as jnp
from ott.geometry.geometry import Geometry
import numpy as np


@pytest.fixture()
def geom_a():
    np.random.seed(0)
    n = 20  # number of points in the first distribution
    sig = 1  # std of first distribution

    phi = np.arange(n)[:, None]
    xs = phi + sig * np.random.randn(n, 1)

    return Geometry(cost_matrix=jnp.asarray(ot.dist(xs)))


@pytest.fixture()
def geom_b():
    np.random.seed(1)
    n = 20  # number of points in the first distribution
    n2 = 30  # number of points in the second distribution
    sig = 1  # std of first distribution
    sig2 = 0.1  # std of second distribution

    np.vstack((np.ones((n // 2, 1)), 0 * np.ones((n // 2, 1)))) + sig2 * np.random.randn(n, 1)
    phi2 = np.arange(n2)[:, None]
    xt = phi2 + sig * np.random.randn(n2, 1)

    return Geometry(cost_matrix=jnp.asarray(ot.dist(xt)))


@pytest.fixture()
def geom_ab():
    np.random.seed(2)
    n = 20  # number of points in the first distribution
    n2 = 30  # number of points in the second distribution
    sig = 1  # std of first distribution
    sig2 = 0.1  # std of second distribution

    phi = np.arange(n)[:, None]
    phi + sig * np.random.randn(n, 1)
    ys = np.vstack((np.ones((n // 2, 1)), 0 * np.ones((n // 2, 1)))) + sig2 * np.random.randn(n, 1)

    phi2 = np.arange(n2)[:, None]
    phi2 + sig * np.random.randn(n2, 1)
    yt = np.vstack((np.ones((n2 // 2, 1)), 0 * np.ones((n2 // 2, 1)))) + sig2 * np.random.randn(n2, 1)
    yt = yt[::-1, :]

    return Geometry(cost_matrix=jnp.asarray(ot.dist(ys, yt)))
