from math import cos, sin
from typing import List, Tuple, Optional

from scipy.sparse import csr_matrix
import pandas as pd
import pytest

from jax.config import config
import numpy as np
import numpy.typing as npt

from anndata import AnnData
import scanpy as sc
import anndata as ad

from tests._utils import Geom_t

ANGLES = (0, 30, 60)


config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

_gt_temporal_adata = sc.read("tests/data/moscot_temporal_tests.h5ad")


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
    return AnnData(X=np.asarray(x, dtype=float), obsm={"X_pca": pc}, dtype=float)


@pytest.fixture()
def adata_y(y: Geom_t) -> AnnData:
    rng = np.random.RandomState(44)
    pc = rng.normal(size=(len(y), 4))
    return AnnData(X=np.asarray(y, dtype=float), obsm={"X_pca": pc}, dtype=float)


@pytest.fixture()
def adata_time() -> AnnData:
    rng = np.random.RandomState(42)
    adatas = [AnnData(X=csr_matrix(rng.normal(size=(96, 60)))) for _ in range(3)]
    adata = adatas[0].concatenate(*adatas[1:], batch_key="time")
    adata.obs["time"] = pd.to_numeric(adata.obs["time"])
    adata.obs["batch"] = rng.choice((0, 1, 2), len(adata))
    adata.obs["left_marginals"] = np.ones(len(adata))
    adata.obs["right_marginals"] = np.ones(len(adata))
    # three genes from mouse/human prliferation/apoptosis
    genes = ["ANLN", "ANP32E", "ATAD2", "Mcm4", "Smc4", "Gtse1", "ADD1", "AIFM3", "ANKH", "Ercc5", "Serpinb5", "Inhbb"]
    adata.var.index = ["gene_" + el if i > 11 else genes[i] for i, el in enumerate(adata.var.index)]
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


@pytest.fixture()
def gt_temporal_adata() -> AnnData:
    return _gt_temporal_adata.copy()


@pytest.fixture()
def adata_space_rotate() -> AnnData:
    grid = _make_grid(10)
    adatas = _make_adata(grid, n=3)
    for adata, angle in zip(adatas, ANGLES):
        theta = np.deg2rad(angle)
        rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        adata.obsm["spatial"] = np.dot(adata.obsm["spatial"], rot)

    adata = ad.concat(adatas, label="batch")
    adata.uns["spatial"] = {}
    adata.obs_names_make_unique()
    return adata


@pytest.fixture()
def adata_mapping() -> AnnData:
    grid = _make_grid(10)
    adataref, adata1, adata2 = _make_adata(grid, n=3)
    sc.pp.pca(adataref)

    adata = ad.concat([adataref, adata1, adata2], label="batch", join="outer")
    adata.obs_names_make_unique()
    return adata


def _make_grid(grid_size: int) -> npt.ArrayLike:
    xlimits = ylimits = [0, 10]
    x1s = np.linspace(*xlimits, num=grid_size)
    x2s = np.linspace(*ylimits, num=grid_size)
    X1, X2 = np.meshgrid(x1s, x2s)
    X_orig_single = np.vstack([X1.ravel(), X2.ravel()]).T
    return X_orig_single


def _make_adata(grid: npt.ArrayLike, n: int) -> List[AnnData]:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 60))
    adatas = [AnnData(X=csr_matrix(X), obsm={"spatial": grid.copy()}) for _ in range(n)]
    return adatas


def _adata_spatial_split(adata: AnnData) -> Tuple[AnnData, AnnData]:
    adata_ref = adata[adata.obs.batch == "0"].copy()
    adata_ref.obsm.pop("spatial")
    adata_sp = adata[adata.obs.batch != "0"].copy()
    return adata_ref, adata_sp
