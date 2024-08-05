from math import cos, sin
from typing import Literal, Optional, Tuple, Union

import pytest

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import config
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt

import anndata as ad
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
    return AnnData(X=np.asarray(x, dtype=float), obsm={"X_pca": pc})


@pytest.fixture()
def adata_y(y: Geom_t) -> AnnData:
    rng = np.random.RandomState(44)
    pc = rng.normal(size=(len(y), 4))
    return AnnData(X=np.asarray(y, dtype=float), obsm={"X_pca": pc})


def creat_prob(n: int, *, uniform: bool = False, seed: Optional[int] = None) -> Geom_t:
    rng = np.random.RandomState(seed)
    a = np.ones((n,)) if uniform else np.abs(rng.normal(size=(n,)))
    a /= np.sum(a)
    return jnp.asarray(a)


@pytest.fixture()
def adata_time() -> AnnData:
    rng = np.random.RandomState(42)

    adatas = [
        AnnData(
            X=csr_matrix(rng.normal(size=(96, 60))),
            obs={
                "left_marginals_balanced": creat_prob(96, seed=42),
                "right_marginals_balanced": creat_prob(96, seed=42),
            },
        )
        for _ in range(3)
    ]
    adata = ad.concat(adatas, label="time", index_unique="-")
    adata.obs["time"] = pd.to_numeric(adata.obs["time"]).astype("category")
    adata.obs["batch"] = rng.choice((0, 1, 2), len(adata))
    adata.obs["left_marginals_unbalanced"] = np.ones(len(adata))
    adata.obs["right_marginals_unbalanced"] = np.ones(len(adata))
    adata.obs["celltype"] = rng.choice(["A", "B", "C"], size=len(adata))
    # genes from mouse/human proliferation/apoptosis
    genes = ["ANLN", "ANP32E", "ATAD2", "Mcm4", "Smc4", "Gtse1", "ADD1", "AIFM3", "ANKH", "Ercc5", "Serpinb5", "Inhbb"]
    # genes which are transcription factors, 3 from drosophila, 2 from human, 1 from mouse
    genes += ["Cf2", "Dlip3", "Dref", "KLF12", "ZNF143", "Zic5"]
    adata.var.index = ["gene_" + el if i > len(genes) - 1 else genes[i] for i, el in enumerate(adata.var.index)]
    adata.obsm["X_umap"] = rng.randn(len(adata), 2)
    sc.pp.pca(adata)
    return adata


@pytest.fixture()
def gt_temporal_adata() -> AnnData:
    adata = _gt_temporal_adata.copy()
    # TODO(michalk8): remove both lines once data has been regenerated
    adata.obs["day"] = pd.to_numeric(adata.obs["day"]).astype("category")
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

    adata = ad.concat(adatas, label="batch", index_unique="-")
    adata.obs["celltype"] = rng.choice(["A", "B", "C"], size=len(adata))
    adata.uns["spatial"] = {}
    sc.pp.pca(adata)
    return adata


@pytest.fixture()
def adata_mapping() -> AnnData:
    grid = _make_grid(10)
    adataref, adata1, adata2 = _make_adata(grid, n=3, seed=17, cat_key="covariate", num_categories=3)
    sc.pp.pca(adataref, n_comps=30)
    return ad.concat([adataref, adata1, adata2], label="batch", join="outer", index_unique="-")


@pytest.fixture()
def adata_translation() -> AnnData:
    rng = np.random.RandomState(31)
    adatas = [AnnData(X=csr_matrix(rng.normal(size=(100, 60)))) for _ in range(3)]
    adata = ad.concat(adatas, label="batch", index_unique="-")
    adata.obs["celltype"] = rng.choice(["A", "B", "C"], size=len(adata))
    adata.obs["celltype"] = adata.obs["celltype"].astype("category")
    adata.layers["counts"] = adata.X.toarray()
    sc.pp.pca(adata)
    return adata


@pytest.fixture()
def adata_translation_split(adata_translation) -> Tuple[AnnData, AnnData]:
    rng = np.random.RandomState(15)
    adata_src = adata_translation[adata_translation.obs.batch != "0"].copy()
    adata_tgt = adata_translation[adata_translation.obs.batch == "0"].copy()
    adata_src.obsm["emb_src"] = rng.normal(size=(adata_src.shape[0], 5))
    adata_tgt.obsm["emb_tgt"] = rng.normal(size=(adata_tgt.shape[0], 15))
    return adata_src, adata_tgt


@pytest.fixture()
def adata_anno(
    problem_kind: Literal["temporal", "cross_modality", "alignment", "mapping"],
) -> Union[AnnData, Tuple[AnnData, AnnData]]:
    rng = np.random.RandomState(31)
    adata_src = AnnData(X=csr_matrix(rng.normal(size=(10, 60))))
    rng_src = rng.choice(["A", "B", "C"], size=5).tolist()
    adata_src.obs["celltype1"] = ["C", "C", "A", "B", "B"] + rng_src
    adata_src.obs["celltype1"] = adata_src.obs["celltype1"].astype("category")
    adata_src.uns["expected_max1"] = ["C", "C", "A", "B", "B"] + rng_src + rng_src
    adata_src.uns["expected_sum1"] = ["C", "C", "B", "B", "B"] + rng_src + rng_src

    adata_tgt = AnnData(X=csr_matrix(rng.normal(size=(15, 60))))
    rng_tgt = rng.choice(["A", "B", "C"], size=5).tolist()
    adata_tgt.obs["celltype2"] = ["C", "C", "A", "B", "B"] + rng_tgt + rng_tgt
    adata_tgt.obs["celltype2"] = adata_tgt.obs["celltype2"].astype("category")
    adata_tgt.uns["expected_max2"] = ["C", "C", "A", "B", "B"] + rng_tgt
    adata_tgt.uns["expected_sum2"] = ["C", "C", "B", "B", "B"] + rng_tgt

    if problem_kind == "cross_modality":
        adata_src.obs["batch"] = "0"
        adata_tgt.obs["batch"] = "1"
        adata_src.obsm["emb_src"] = rng.normal(size=(adata_src.shape[0], 5))
        adata_tgt.obsm["emb_tgt"] = rng.normal(size=(adata_tgt.shape[0], 15))
        sc.pp.pca(adata_src)
        sc.pp.pca(adata_tgt)
        return adata_src, adata_tgt
    if problem_kind == "mapping":
        adata_src.obs["batch"] = "0"
        adata_tgt.obs["batch"] = "1"
        sc.pp.pca(adata_src)
        sc.pp.pca(adata_tgt)
        adata_tgt.obsm["spatial"] = rng.normal(size=(adata_tgt.n_obs, 2))
        return adata_src, adata_tgt
    if problem_kind == "alignment":
        adata_src.obsm["spatial"] = rng.normal(size=(adata_src.n_obs, 2))
        adata_tgt.obsm["spatial"] = rng.normal(size=(adata_tgt.n_obs, 2))
    key = "day" if problem_kind == "temporal" else "batch"
    adatas = [adata_src, adata_tgt]
    adata = ad.concat(adatas, join="outer", label=key, index_unique="-", uns_merge="unique")
    adata.obs[key] = (pd.to_numeric(adata.obs[key]) if key == "day" else adata.obs[key]).astype("category")
    adata.layers["counts"] = adata.X.toarray()
    sc.pp.pca(adata)
    return adata


@pytest.fixture()
def gt_tm_annotation() -> np.ndarray:
    tm = np.zeros((10, 15))
    for i in range(10):
        tm[i][i] = 1
    for i in range(10, 15):
        tm[i - 5][i] = 1
    for j in range(2, 5):
        for i in range(2, 5):
            tm[i][j] = 0.3 if i != j else 0.4
    return tm
