import pytest

import numpy as np

from anndata import AnnData

from tests._utils import _make_grid


@pytest.fixture()
def adata_spatio_temporal(adata_time: AnnData) -> AnnData:
    _, t_unique_counts = np.unique(adata_time.obs["time"], return_counts=True)
    print(t_unique_counts)
    grids = []
    for i, c in enumerate(t_unique_counts):
        grids += [_make_grid(c)[:c, :] + i]
    print(
        "grids shapes are ",
        grids[0].shape,
        grids[1].shape,
        grids[2].shape,
    )
    adata_time.obsm["spatial"] = np.concatenate(grids, axis=0)

    return adata_time
