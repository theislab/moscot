from math import cos, sin
from typing import Optional, Tuple

import pytest

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.config import config
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt

import scanpy as sc
from anndata import AnnData

from tests._utils import Geom_t, _make_adata, _make_grid

ANGLES = (0, 30, 60)


# TODO(michalk8): consider passing this via env
config.update("jax_enable_x64", True)


_gt_temporal_adata = sc.read("tests/data/moscot_temporal_tests.h5ad")


def pytest_sessionstart() -> None:
    sc.pl.set_rcParams_defaults()
    sc.set_figure_params(dpi=40, color_map="viridis")


@pytest.fixture(autouse=True)
def _close_figure():
    # prevent `RuntimeWarning: More than 20 figures have been opened.`
    yield
    plt.close()


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
def xy() -> Tuple[Geom_t, Geom_t]:
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
    adatas = [AnnData(X=csr_matrix(rng.normal(size=(96, 60))), dtype=float) for _ in range(3)]
    adata = adatas[0].concatenate(*adatas[1:], batch_key="time")
    adata.obs["time"] = pd.to_numeric(adata.obs["time"])
    adata.obs["batch"] = rng.choice((0, 1, 2), len(adata))
    adata.obs["left_marginals"] = np.ones(len(adata))
    adata.obs["right_marginals"] = np.ones(len(adata))
    adata.obs["celltype"] = rng.choice(["A", "B", "C"], size=len(adata))
    # genes from mouse/human proliferation/apoptosis
    genes = ["ANLN", "ANP32E", "ATAD2", "Mcm4", "Smc4", "Gtse1", "ADD1", "AIFM3", "ANKH", "Ercc5", "Serpinb5", "Inhbb"]
    # genes which are transcription factors, 3 from drosophila, 2 from human, 1 from mouse
    genes += ["Cf2", "Dlip3", "Dref", "KLF12", "ZNF143", "Zic5"]
    adata.var.index = ["gene_" + el if i > len(genes) - 1 else genes[i] for i, el in enumerate(adata.var.index)]
    adata.obsm["X_umap"] = rng.randn(len(adata), 2)
    sc.pp.pca(adata)
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
    adata = _gt_temporal_adata.copy()
    adata.obs_names_make_unique()
    return adata


@pytest.fixture()
def adata_space_rotate() -> AnnData:
    rng = np.random.RandomState(31)
    grid = _make_grid(10)
    adatas = _make_adata(grid, n=len(ANGLES), seed=32)
    for adata, angle in zip(adatas, ANGLES):
        theta = np.deg2rad(angle)
        rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        adata.obsm["spatial"] = np.dot(adata.obsm["spatial"], rot)

    adata = adatas[0].concatenate(*adatas[1:], batch_key="batch")
    adata.obs["celltype"] = rng.choice(["A", "B", "C"], size=len(adata))
    adata.uns["spatial"] = {}
    adata.obs_names_make_unique()
    sc.pp.pca(adata)
    return adata


@pytest.fixture()
def adata_mapping() -> AnnData:
    grid = _make_grid(10)
    adataref, adata1, adata2 = _make_adata(grid, n=3, seed=17)
    sc.pp.pca(adataref, n_comps=30)

    adata = adataref.concatenate(adata1, adata2, batch_key="batch", join="outer")
    adata.obs_names_make_unique()
    return adata

@pytest.fixture()
def adata_translation() -> AnnData:
    rng = np.random.RandomState(31)
    adatas = [AnnData(X=csr_matrix(rng.normal(size=(100, 60))), dtype=float) for _ in range(3)]
    adata = adatas[0].concatenate(*adatas[1:], batch_key="batch")
    adata.obs["celltype"] = rng.choice(["A", "B", "C"], size=len(adata))
    adata.obs["celltype"] = adata.obs["celltype"].astype("category")
    adata.obs_names_make_unique()
    sc.pp.pca(adata)
    return adata