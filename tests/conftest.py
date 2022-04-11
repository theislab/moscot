from typing import Tuple, Union, Optional

import scipy
import pandas as pd
import pytest

from jax.config import config

from anndata import AnnData

config.update("jax_enable_x64", True)
from sklearn.metrics import pairwise_distances

from jax import numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

Geom_t = Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]
RTOL = 1e-6
ATOL = 1e-6


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
    rng = np.random.RandomState(43)
    pc = rng.normal(size=(len(x), 4))
    return AnnData(X=np.asarray(x), obsm={"X_pca": pc})


@pytest.fixture()
def adata_y(y: Geom_t) -> AnnData:
    rng = np.random.RandomState(44)
    pc = rng.normal(size=(len(y), 4))
    return AnnData(X=np.asarray(y), obsm={"X_pca": pc})


@pytest.fixture()
def adata_time() -> AnnData:
    rng = np.random.RandomState(42)
    adatas = [AnnData(X=rng.normal(size=(96, 30))) for _ in range(3)]
    adata = adatas[0].concatenate(*adatas[1:], batch_key="time")
    adata.obs["time"] = pd.to_numeric(adata.obs["time"])
    # three genes from mouse/human prliferation/apoptosis
    genes = ["ANLN", "ANP32E", "ATAD2", "Mcm4", "Smc4", "Gtse1", "ADD1", "AIFM3", "ANKH", "Ercc5", "Serpinb5", "Inhbb"]
    adata.var.index = ["gene_" + el if i > 11 else genes[i] for i, el in enumerate(adata.var.index)]
    return adata


@pytest.fixture()
def adata_time_cell_type(adata_time: AnnData) -> AnnData:
    rng = np.random.RandomState(42)
    adata_time.obs["cell_type"] = rng.choice(["cell_A", "cell_B", "cell_C"], size=len(adata_time))
    return adata_time


@pytest.fixture()
def adata_time_barcodes(adata_time: AnnData) -> AnnData:
    rng = np.random.RandomState(42)
    adata_time.obsm["barcodes"] = rng.randn(len(adata_time), 30)
    return adata_time


@pytest.fixture()
def adata_time_trees() -> AnnData:  # TODO(@MUCDK) create
    pass


@pytest.fixture()
def adata_time_custom_cost_xy(adata_time: AnnData) -> AnnData:
    rng = np.random.RandomState(42)
    cost_m1 = np.abs(rng.randn(96, 96))
    cost_m2 = np.abs(rng.randn(96, 96))
    cost_m3 = np.abs(rng.randn(96, 96))
    adata_time.obsp["cost_matrices"] = scipy.sparse.csr_matrix(scipy.linalg.block_diag(cost_m1, cost_m2, cost_m3))
    return adata_time


@pytest.fixture()
def random_transport_matrix_adata_time(adata_time_cell_type: AnnData) -> np.ndarray:
    rng = np.random.RandomState(42)
    adata = adata_time_cell_type
    dim_0 = adata[adata.obs["time"] == 0].n_obs
    dim_1 = adata[adata.obs["time"] == 1].n_obs
    t_matrix = np.abs(rng.randn(dim_0, dim_1))
    return t_matrix / t_matrix.sum()


@pytest.fixture()
def adata_with_cost_matrix(adata_x: Geom_t, adata_y: Geom_t):
    adata = adata_x.concatenate(adata_y, batch_key="batch")
    C = pairwise_distances(adata_x.obsm["X_pca"], adata_y.obsm["X_pca"]) ** 2
    adata.obs["batch"] = pd.to_numeric(adata.obs["batch"])
    adata.uns[0] = C / C.mean()  # TODO(@MUCDK) make a callback function and replace this part
    return adata


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
