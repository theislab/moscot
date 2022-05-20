from typing import Optional

import pytest

import numpy as np

from anndata import AnnData
from scipy.sparse.linalg import LinearOperator
from tests._utils import MockSolverOutput, CompoundProblemWithMixin


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
        problem[10, 10.5]._solution = MockSolverOutput(gt_temporal_adata.uns["tmap_10_105"])
        problem[10.5, 11]._solution = MockSolverOutput(gt_temporal_adata.uns["tmap_105_11"])
        problem[10, 11]._solution = MockSolverOutput(gt_temporal_adata.uns["tmap_10_11"])

        tmap = problem._interpolate_transport(10, 11, forward=forward, scale_by_marginals=True)

        assert isinstance(tmap, LinearOperator)
        # TODO(@MUCDK) add regression test after discussing with @giovp what this function should be
        # doing / it is more generic
