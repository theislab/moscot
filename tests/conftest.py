from typing import Tuple, Optional

import ot
import pytest

from jax.config import config

config.update("jax_enable_x64", True)

from jax import numpy as jnp  # noqa: E402
from ott.geometry.geometry import Geometry  # noqa: E402
import numpy as np  # noqa: E402


@pytest.fixture()
def geom_a() -> Geometry:
    np.random.seed(0)
    n = 20  # number of points in the first distribution
    sig = 1  # std of first distribution

    phi = np.arange(n)[:, None]
    xs = phi + sig * np.random.randn(n, 1)

    return Geometry(cost_matrix=jnp.asarray(ot.dist(xs)))


@pytest.fixture()
def geom_b() -> Geometry:
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
def geom_ab() -> Geometry:
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


def create_marginals(
    n: int, m: int, *, uniform: bool = False, seed: Optional[int] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    np.random.seed(seed)
    if uniform:
        a, b = np.ones((n,)), np.ones((m,))
    else:
        a = np.abs(np.random.normal(size=(n,)))
        b = np.abs(np.random.normal(size=(m,)))
    a /= np.sum(a)
    b /= np.sum(b)
    a = jnp.asarray(a)
    b = jnp.asarray(b)

    return a, b
