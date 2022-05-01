from math import cos, sin
from typing import List

import pytest

import numpy as np

from anndata import AnnData
import anndata as ad


@pytest.fixture()
def adata_space_rotate(adatas_space: List[AnnData]) -> AnnData:
    adatas = adatas_space.copy()
    for adata, angle in zip(adatas, [0, 25, 50]):
        theta = np.deg2rad(angle)
        rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        adata.obsm["spatial"] = np.dot(adata.obsm["spatial"], rot)

    adata.obs_names_make_unique()
    adata = ad.concat(adatas, label="batch")
    adata.obs_names_make_unique()

    return adata
