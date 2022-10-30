from typing import Optional

from scipy.sparse.linalg import LinearOperator
import pandas as pd
import pytest

import numpy as np

from anndata import AnnData

from tests._utils import ATOL, RTOL, MockSolverOutput, CompoundProblemWithMixin


class TestBaseAnalysisMixin:
    @pytest.mark.parametrize("n_samples", [10, 42])
    @pytest.mark.parametrize("account_for_unbalancedness", [True, False])
    @pytest.mark.parametrize("interpolation_parameter", [None, 0.1, 5])
    def test_sample_from_tmap_pipeline(
        self,
        gt_temporal_adata: AnnData,
        n_samples: int,
        account_for_unbalancedness: bool,
        interpolation_parameter: Optional[float],
    ):
        source_dim = len(gt_temporal_adata[gt_temporal_adata.obs["day"] == 10])
        target_dim = len(gt_temporal_adata[gt_temporal_adata.obs["day"] == 10.5])
        problem = CompoundProblemWithMixin(gt_temporal_adata)
        problem = problem.prepare("day", subset=[(10, 10.5)], policy="sequential", callback="local-pca")
        problem[10, 10.5]._solution = MockSolverOutput(gt_temporal_adata.uns["tmap_10_105"])

        if interpolation_parameter is not None and not 0 <= interpolation_parameter <= 1:
            with np.testing.assert_raises(ValueError):
                problem._sample_from_tmap(
                    10,
                    10.5,
                    n_samples,
                    source_dim=source_dim,
                    target_dim=target_dim,
                    account_for_unbalancedness=account_for_unbalancedness,
                    interpolation_parameter=interpolation_parameter,
                )
        elif interpolation_parameter is None and account_for_unbalancedness:
            with np.testing.assert_raises(ValueError):
                problem._sample_from_tmap(
                    10,
                    10.5,
                    n_samples,
                    source_dim=source_dim,
                    target_dim=target_dim,
                    account_for_unbalancedness=account_for_unbalancedness,
                    interpolation_parameter=interpolation_parameter,
                )
        else:
            result = problem._sample_from_tmap(
                10,
                10.5,
                n_samples,
                source_dim=source_dim,
                target_dim=target_dim,
                account_for_unbalancedness=account_for_unbalancedness,
                interpolation_parameter=interpolation_parameter,
            )
            assert isinstance(result, tuple)
            assert isinstance(result[0], np.ndarray)
            assert isinstance(result[1], list)
            assert isinstance(result[1][0], np.ndarray)
            assert len(np.concatenate(result[1])) == n_samples

    @pytest.mark.parametrize("forward", [True, False])
    @pytest.mark.parametrize("scale_by_marginals", [True, False])
    def test_interpolate_transport(self, gt_temporal_adata: AnnData, forward: bool, scale_by_marginals: bool):
        problem = CompoundProblemWithMixin(gt_temporal_adata)
        problem = problem.prepare(
            "day", subset=[(10, 10.5), (10.5, 11), (10, 11)], policy="explicit", callback="local-pca"
        )
        problem[(10.0, 10.5)]._solution = MockSolverOutput(gt_temporal_adata.uns["tmap_10_105"])
        problem[(10.5, 11.0)]._solution = MockSolverOutput(gt_temporal_adata.uns["tmap_105_11"])
        problem[(10.0, 11.0)]._solution = MockSolverOutput(gt_temporal_adata.uns["tmap_10_11"])
        tmap = problem._interpolate_transport(
            [(10, 11)], forward=forward, scale_by_marginals=True, explicit_steps=[(10.0, 11.0)]
        )

        assert isinstance(tmap, LinearOperator)
        # TODO(@MUCDK) add regression test after discussing with @giovp what this function should be
        # doing / it is more generic

    def test_cell_transition_aggregation_cell_forward(self, gt_temporal_adata: AnnData):
        # the method used in this test does the same but has to instantiate the transport matrix
        config = gt_temporal_adata.uns
        config["key"]
        key_1 = config["key_1"]
        key_2 = config["key_2"]
        config["key_3"]
        problem = CompoundProblemWithMixin(gt_temporal_adata)
        problem = problem.prepare("day", subset=[(10, 10.5)], policy="explicit", callback="local-pca")
        assert set(problem.problems.keys()) == {(key_1, key_2)}
        problem[key_1, key_2]._solution = MockSolverOutput(gt_temporal_adata.uns["tmap_10_105"])

        ctr = problem._cell_transition(
            key="day",
            source=10,
            target=10.5,
            source_groups="cell_type",
            target_groups="cell_type",
            forward=True,
            aggregation_mode="cell",
        )

        adata_early = gt_temporal_adata[gt_temporal_adata.obs["day"] == 10]
        adata_late = gt_temporal_adata[gt_temporal_adata.obs["day"] == 10.5]

        transition_matrix_indexed = pd.DataFrame(
            index=adata_early.obs.index, columns=adata_late.obs.index, data=gt_temporal_adata.uns["tmap_10_105"]
        )
        unique_cell_types_late = adata_late.obs["cell_type"].cat.categories
        df_res = pd.DataFrame(index=adata_early.obs.index)
        for ct in unique_cell_types_late:
            cols_cell_type = adata_late[adata_late.obs["cell_type"] == ct].obs.index
            df_res[ct] = transition_matrix_indexed[cols_cell_type].sum(axis=1)

        df_res = df_res.div(df_res.sum(axis=1), axis=0)

        ctr_ordered = ctr.sort_index().sort_index(1)
        df_res_ordered = df_res.sort_index().sort_index(1)
        np.testing.assert_allclose(
            ctr_ordered.values.astype(float), df_res_ordered.values.astype(float), rtol=RTOL, atol=ATOL
        )

    def test_cell_transition_aggregation_cell_backward(self, gt_temporal_adata: AnnData):
        # the method used in this test does the same but has to instantiate the transport matrix
        config = gt_temporal_adata.uns
        config["key"]
        key_1 = config["key_1"]
        key_2 = config["key_2"]
        problem = CompoundProblemWithMixin(gt_temporal_adata)
        problem = problem.prepare("day", subset=[(10, 10.5)], policy="explicit", callback="local-pca")
        assert set(problem.problems.keys()) == {(key_1, key_2)}
        problem[key_1, key_2]._solution = MockSolverOutput(gt_temporal_adata.uns["tmap_10_105"])

        ctr = problem._cell_transition(
            key="day",
            source=10,
            target=10.5,
            source_groups="cell_type",
            target_groups="cell_type",
            forward=False,
            aggregation_mode="cell",
        )

        adata_early = gt_temporal_adata[gt_temporal_adata.obs["day"] == 10]
        adata_late = gt_temporal_adata[gt_temporal_adata.obs["day"] == 10.5]

        transition_matrix_indexed = pd.DataFrame(
            index=adata_early.obs.index, columns=adata_late.obs.index, data=gt_temporal_adata.uns["tmap_10_105"]
        )
        unique_cell_types_early = adata_early.obs["cell_type"].cat.categories
        df_res = pd.DataFrame(columns=adata_late.obs.index)
        for ct in unique_cell_types_early:
            rows_cell_type = adata_early[adata_early.obs["cell_type"] == ct].obs.index
            df_res.loc[ct] = transition_matrix_indexed.loc[rows_cell_type].sum(axis=0)

        df_res = df_res.div(df_res.sum(axis=0), axis=1)

        ctr_ordered = ctr.sort_index().sort_index(1)
        df_res_ordered = df_res.sort_index().sort_index(1)
        np.testing.assert_allclose(
            ctr_ordered.values.astype(float), df_res_ordered.values.astype(float), rtol=RTOL, atol=ATOL
        )
