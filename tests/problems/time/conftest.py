import scipy
import pytest

import numpy as np

from anndata import AnnData

from moscot.datasets._utils import _get_random_trees


@pytest.fixture()
def adata_time_trees(adata_time: AnnData) -> AnnData:
    trees = _get_random_trees(
        n_leaves=96, n_trees=3, leaf_names=[list(adata_time[adata_time.obs.time == i].obs.index) for i in range(3)]
    )
    adata_time.uns["trees"] = {0: trees[0], 1: trees[1], 2: trees[2]}
    return adata_time


@pytest.fixture()
def adata_time_custom_cost_xy(adata_time: AnnData) -> AnnData:
    rng = np.random.RandomState(42)
    cost_m1 = np.abs(rng.randn(96, 96))
    cost_m2 = np.abs(rng.randn(96, 96))
    cost_m3 = np.abs(rng.randn(96, 96))
    adata_time.obsp["cost_matrices"] = scipy.sparse.csr_matrix(scipy.linalg.block_diag(cost_m1, cost_m2, cost_m3))
    return adata_time


@pytest.fixture()
def adata_time_barcodes(adata_time: AnnData) -> AnnData:
    rng = np.random.RandomState(42)
    adata_time.obsm["barcodes"] = rng.randn(len(adata_time), 30)
    return adata_time


@pytest.fixture()
def adata_time_marginal_estimations(adata_time: AnnData) -> AnnData:
    rng = np.random.RandomState(42)
    adata_time.obs["proliferation"] = rng.randn(len(adata_time))
    adata_time.obs["apoptosis"] = rng.randn(len(adata_time))
    return adata_time
