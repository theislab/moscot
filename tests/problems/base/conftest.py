import pytest

import numpy as np

import anndata as ad

from moscot.backends.ott._jax_data import JaxSampler


@pytest.fixture()
def sampler_no_conditions(adata_time: ad.AnnData) -> JaxSampler:
    dist_0 = adata_time[adata_time.obs["time"] == 0].obsm["X_pca"].copy()
    dist_1 = adata_time[adata_time.obs["time"] == 1].obsm["X_pca"].copy()
    dist_2 = adata_time[adata_time.obs["time"] == 2].obsm["X_pca"].copy()
    a_0 = np.ones((dist_0.shape[0], 1))
    a_1 = np.ones((dist_1.shape[0], 1))
    a_2 = np.ones((dist_2.shape[0], 1))

    distributions = [dist_0, dist_1, dist_2]
    a = [a_0, a_1, a_2]
    b = a
    policy_pairs = [(0, 1), (1, 2)]
    sample_to_idx = {0: 0, 1: 1, 2: 2}

    return JaxSampler(distributions, policy_pairs=policy_pairs, conditions=None, sample_to_idx=sample_to_idx, a=a, b=b)


@pytest.fixture()
def sampler_with_conditions(adata_time: ad.AnnData) -> JaxSampler:
    dist_0 = adata_time[adata_time.obs["time"] == 0].obsm["X_pca"].copy()
    dist_1 = adata_time[adata_time.obs["time"] == 1].obsm["X_pca"].copy()
    dist_2 = adata_time[adata_time.obs["time"] == 2].obsm["X_pca"].copy()
    a_0 = np.ones((dist_0.shape[0], 1))
    a_1 = np.ones((dist_1.shape[0], 1))
    a_2 = np.ones((dist_2.shape[0], 1))

    cond_0 = adata_time[adata_time.obs["time"] == 0].obs["time"]
    cond_1 = adata_time[adata_time.obs["time"] == 1].obs["time"]
    cond_2 = adata_time[adata_time.obs["time"] == 2].obs["time"]

    distributions = [dist_0, dist_1, dist_2]
    a = [a_0, a_1, a_2]
    b = a
    policy_pairs = [(0, 1), (1, 2)]
    sample_to_idx = {0: 0, 1: 1, 2: 2}
    conditions = [cond_0, cond_1, cond_2]

    return JaxSampler(
        distributions, policy_pairs=policy_pairs, conditions=conditions, sample_to_idx=sample_to_idx, a=a, b=b
    )
