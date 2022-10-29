from sklearn.metrics import pairwise_distances
import pytest

import numpy as np

import scanpy as sc

from moscot.costs import cost_to_obsp  # type: ignore[attr-defined]
from moscot.datasets import simulate_data
from moscot.problems.time import TemporalProblem  # type: ignore[attr-defined]


class TestCost:
    @pytest.mark.fast()
    def test_obs_to_cost_pipeline(self) -> None:
        adata = simulate_data(n_distributions=2, key="batch")
        sc.pp.pca(adata)
        adata.obs["day"] = np.random.choice([0, 1, 2], len(adata))

        obs_0 = adata[adata.obs["day"] == 0].obs_names
        obs_1 = adata[adata.obs["day"] == 1].obs_names
        obs_2 = adata[adata.obs["day"] == 1].obs_names
        cell_embedding_0 = adata[obs_0].obsm["X_pca"]
        cell_embedding_1 = adata[obs_1].obsm["X_pca"]
        cell_embedding_2 = adata[obs_2].obsm["X_pca"]

        cost_matrix01 = pairwise_distances(cell_embedding_0, cell_embedding_1, metric="sqeuclidean")
        adata = cost_to_obsp(adata, "obsp_added", cost_matrix01, obs_0, obs_1)
        cost_matrix12 = pairwise_distances(cell_embedding_1, cell_embedding_2, metric="sqeuclidean")
        adata = cost_to_obsp(adata, "obsp_added", cost_matrix12, obs_1, obs_2)

        tp = TemporalProblem(adata)
        tp = tp.prepare(time_key="day", callback="cost-matrix", callback_kwargs={"key": "obsp_added"})

        np.testing.assert_allclose(tp[0, 1].xy.data, cost_matrix01, atol=1e-7)
        np.testing.assert_allclose(tp[1, 2].xy.data, cost_matrix12, atol=1e-7)

    def test_obs_to_cost_regression(self) -> None:
        adata = simulate_data(n_distributions=2, key="batch")
        sc.pp.pca(adata)
        adata.obs["day"] = np.random.choice([0, 1], len(adata))

        tp = TemporalProblem(adata)
        tp = tp.prepare(time_key="day", joint_attr="X_pca")
        tp = tp.solve()

        obs_0 = adata[adata.obs["day"] == 0].obs_names
        obs_1 = adata[adata.obs["day"] == 1].obs_names
        cell_embedding_0 = adata[obs_0].obsm["X_pca"]
        cell_embedding_1 = adata[obs_1].obsm["X_pca"]
        cost_matrix = pairwise_distances(cell_embedding_0, cell_embedding_1, metric="sqeuclidean")

        adata = cost_to_obsp(adata, "obsp_added", cost_matrix, obs_0, obs_1)
        tp2 = TemporalProblem(adata)
        tp2 = tp2.prepare(time_key="day", callback="cost-matrix", callback_kwargs={"key": "obsp_added"})
        tp2 = tp2.solve()

        np.testing.assert_allclose(  # type: ignore[call-overload]
            tp2[0, 1].solution.transport_matrix, tp[0, 1].solution.transport_matrix, tolerance=1e-7
        )
