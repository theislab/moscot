from math import cos, sin
from typing import List

from scipy.sparse import csr_matrix
import pytest

import numpy as np
import numpy.typing as npt

from anndata import AnnData
import anndata as ad

ANGLES = [0, 30, 60]


@pytest.fixture()
def adata_space_rotate() -> AnnData:
    grid = _make_grid(10)
    adatas = _make_adata(grid)
    for adata, angle in zip(adatas, ANGLES):
        theta = np.deg2rad(angle)
        rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        adata.obsm["spatial"] = np.dot(adata.obsm["spatial"], rot)

    adata = ad.concat(adatas, label="batch")
    adata.obs_names_make_unique()
    return adata


def _make_grid(grid_size: int) -> npt.ArrayLike:
    xlimits = ylimits = [0, 10]
    x1s = np.linspace(*xlimits, num=grid_size)
    x2s = np.linspace(*ylimits, num=grid_size)
    X1, X2 = np.meshgrid(x1s, x2s)
    X_orig_single = np.vstack([X1.ravel(), X2.ravel()]).T
    return X_orig_single


def _make_adata(grid: npt.ArrayLike) -> List[AnnData]:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 60))
    adatas = [AnnData(X=csr_matrix(X), obsm={"spatial": grid.copy()}) for _ in range(3)]
    return adatas
