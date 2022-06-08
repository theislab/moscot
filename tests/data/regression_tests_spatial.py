from math import cos, sin
from typing import List, Tuple
import pickle

from scipy.sparse import csr_matrix

import numpy as np

from anndata import AnnData
import scanpy as sc
import anndata as ad

from moscot._types import ArrayLike
from moscot.problems.space import MappingProblem, AlignmentProblem  # type: ignore[attr-defined]

ANGLES = [0, 30, 60]


def adata_space_rotate() -> AnnData:
    grid = _make_grid(10)
    adatas = _make_adata(grid, n=3)
    for adata, angle in zip(adatas, ANGLES):
        theta = np.deg2rad(angle)
        rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        adata.obsm["spatial"] = np.dot(adata.obsm["spatial"], rot)

    adata = ad.concat(adatas, label="batch")
    adata.uns["spatial"] = {}
    return adata


def adata_mapping() -> AnnData:
    grid = _make_grid(10)
    adataref, adata1, adata2 = _make_adata(grid, n=3)
    sc.pp.pca(adataref)

    adata = ad.concat([adataref, adata1, adata2], label="batch", join="outer")
    return adata


def _make_grid(grid_size: int) -> ArrayLike:
    x1s = np.linspace(0, 10, num=grid_size)
    x2s = np.linspace(0, 10, num=grid_size)
    X1, X2 = np.meshgrid(x1s, x2s)
    X_orig_single = np.hstack([X1.ravel(), X2.ravel()])
    return X_orig_single


def _make_adata(grid: ArrayLike, n: int) -> List[AnnData]:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 60))
    return [AnnData(X=csr_matrix(X), obsm={"spatial": grid.copy()}, dtype=X.dtype) for _ in range(n)]


def _adata_split(adata: AnnData) -> Tuple[AnnData, AnnData]:
    adataref = adata[adata.obs["batch"] == "0"].copy()
    adataref.obsm.pop("spatial")
    adatasp = adata[adata.obs["batch"] != "0"].copy()
    return adataref, adatasp


def generate_alignment_data() -> None:
    adata = adata_space_rotate()
    ap = AlignmentProblem(adata=adata)
    ap = ap.prepare(batch_key="batch")
    ap = ap.solve(alpha=0.5, epsilon=1)

    with open("alignment_solutions.pkl", "wb") as fname:
        pickle.dump(ap.solutions, fname)


def generate_mapping_data() -> None:
    adata = adata_mapping()
    adataref, adatasp = _adata_split(adata)
    mp = MappingProblem(adataref, adatasp)
    mp = mp.prepare(batch_key="batch", sc_attr={"attr": "X"})
    mp = mp.solve()
    with open("mapping_solutions.pkl", "wb") as fname:
        pickle.dump(mp.solutions, fname)


if __name__ == "__main__":
    generate_alignment_data()
    generate_mapping_data()
