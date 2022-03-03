from typing import Tuple, Union, Optional

import pytest

from jax.config import config

from anndata import AnnData

config.update("jax_enable_x64", True)

from jax import numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

Geom_t = Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]


@pytest.fixture()
def x() -> Geom_t:
    rng = np.random.RandomState(0)
    n = 20  # number of points in the first distribution
    sig = 1  # std of first distribution

    phi = np.arange(n)[:, None]
    xs = phi + sig * rng.randn(n, 1)

    return jnp.asarray(xs)


@pytest.fixture()
def y() -> Geom_t:
    rng = np.random.RandomState(1)
    n2 = 30  # number of points in the second distribution
    sig = 1  # std of first distribution

    phi2 = np.arange(n2)[:, None]
    xt = phi2 + sig * rng.randn(n2, 1)

    return jnp.asarray(xt)


@pytest.fixture()
def xy() -> Geom_t:
    rng = np.random.RandomState(2)
    n = 20  # number of points in the first distribution
    n2 = 30  # number of points in the second distribution
    sig = 1  # std of first distribution
    sig2 = 0.1  # std of second distribution

    phi = np.arange(n)[:, None]
    phi + sig * rng.randn(n, 1)
    ys = np.vstack((np.ones((n // 2, 1)), 0 * np.ones((n // 2, 1)))) + sig2 * rng.randn(n, 1)

    phi2 = np.arange(n2)[:, None]
    phi2 + sig * rng.randn(n2, 1)
    yt = np.vstack((np.ones((n2 // 2, 1)), 0 * np.ones((n2 // 2, 1)))) + sig2 * rng.randn(n2, 1)
    yt = yt[::-1, :]

    return jnp.asarray(ys), jnp.asarray(yt)


@pytest.fixture()
def ab() -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(42)
    return rng.normal(size=(20, 2)), rng.normal(size=(30, 4))


@pytest.fixture()
def x_cost(x: Geom_t) -> jnp.ndarray:
    return ((x[:, None, :] - x[None, ...]) ** 2).sum(-1)


@pytest.fixture()
def y_cost(y: Geom_t) -> jnp.ndarray:
    return ((y[:, None, :] - y[None, ...]) ** 2).sum(-1)


@pytest.fixture()
def xy_cost(xy: Geom_t) -> jnp.ndarray:
    x, y = xy
    return ((x[:, None, :] - y[None, ...]) ** 2).sum(-1)


@pytest.fixture()
def adata_x(x: Geom_t) -> AnnData:
    return AnnData(X=np.asarray(x))


@pytest.fixture()
def adata_y(y: Geom_t) -> AnnData:
    return AnnData(X=np.asarray(y))


@pytest.fixture()
def adata_xy(xy_cost: jnp.ndarray) -> AnnData:
    return AnnData(X=np.asarray(xy_cost))


def create_marginals(n: int, m: int, *, uniform: bool = False, seed: Optional[int] = None) -> Geom_t:
    rng = np.random.RandomState(seed)
    if uniform:
        a, b = np.ones((n,)), np.ones((m,))
    else:
        a = np.abs(rng.normal(size=(n,)))
        b = np.abs(rng.normal(size=(m,)))
    a /= np.sum(a)
    b /= np.sum(b)

    return jnp.asarray(a), jnp.asarray(b)
