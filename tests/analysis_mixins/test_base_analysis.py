from typing import List, Tuple, Literal, Optional

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
        problem = problem.prepare("day", subset=[(10, 10.5)], policy="sequential", xy_callback="local-pca")
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
            "day", subset=[(10, 10.5), (10.5, 11), (10, 11)], policy="explicit", xy_callback="local-pca"
        )
        problem[(10.0, 10.5)]._solution = MockSolverOutput(gt_temporal_adata.uns["tmap_10_105"])
        problem[(10.5, 11.0)]._solution = MockSolverOutput(gt_temporal_adata.uns["tmap_105_11"])
        problem[(10.0, 11.0)]._solution = MockSolverOutput(gt_temporal_adata.uns["tmap_10_11"])
        tmap = problem._interpolate_transport([(10, 11)], scale_by_marginals=True, explicit_steps=[(10.0, 11.0)])

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
        problem = problem.prepare("day", subset=[(10, 10.5)], policy="explicit", xy_callback="local-pca")
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
        problem = problem.prepare("day", subset=[(10, 10.5)], policy="explicit", xy_callback="local-pca")
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

    @pytest.mark.parametrize("method", ["fischer", "perm_test"])
    def test_compute_feature_correlation(self, adata_time: AnnData, method: Literal["fischer", "perm_test"]):
        key_added = "test"
        rng = np.random.RandomState(42)
        adata_time = adata_time[adata_time.obs["time"].isin((0, 1))].copy()
        n0 = adata_time[adata_time.obs["time"] == 0].n_obs
        n1 = adata_time[adata_time.obs["time"] == 1].n_obs
        tmap = rng.uniform(1e-6, 1, size=(n0, n1))
        tmap /= tmap.sum().sum()
        problem = CompoundProblemWithMixin(adata_time)
        problem = problem.prepare("time", xy_callback="local-pca")
        problem[0, 1]._solution = MockSolverOutput(tmap)

        adata_time.obs[key_added] = np.hstack((np.zeros(n0), problem.pull(start=0, end=1).squeeze()))

        res = problem.compute_feature_correlation(obs_key=key_added, method=method)

        assert isinstance(res, pd.DataFrame)
        assert res.isnull().values.sum() == 0

        assert np.all(res[f"{key_added}_corr"] >= -1.0)
        assert np.all(res[f"{key_added}_corr"] <= 1.0)

        assert np.all(res[f"{key_added}_qval"] >= 0)
        assert np.all(res[f"{key_added}_qval"] <= 1.0)

    @pytest.mark.parametrize("features", [10, None])
    @pytest.mark.parametrize("method", ["fischer", "perm_test"])
    def test_compute_feature_correlation_subset(
        self, adata_time: AnnData, features: Optional[int], method: Literal["fischer", "perm_test"]
    ):
        key_added = "test"
        rng = np.random.RandomState(42)
        adata_time = adata_time[adata_time.obs["time"].isin((0, 1))].copy()
        n0 = adata_time[adata_time.obs["time"] == 0].n_obs
        n1 = adata_time[adata_time.obs["time"] == 1].n_obs
        tmap = rng.uniform(1e-6, 1, size=(n0, n1))
        tmap /= tmap.sum().sum()
        problem = CompoundProblemWithMixin(adata_time)
        problem = problem.prepare("time", xy_callback="local-pca")
        problem[0, 1]._solution = MockSolverOutput(tmap)

        adata_time.obs[key_added] = np.hstack((np.zeros(n0), problem.pull(start=0, end=1).squeeze()))

        if isinstance(features, int):
            features = list(adata_time.var_names)[:features]
            features_validation = features
        else:
            features_validation = list(adata_time.var_names)
        res = problem.compute_feature_correlation(
            obs_key=key_added, annotation={"celltype": ["A"]}, method=method, features=features
        )
        assert isinstance(res, pd.DataFrame)
        assert res.isnull().values.sum() == 0
        assert set(res.index) == set(features_validation)

    @pytest.mark.parametrize(
        "features",
        [
            ("human", ["KLF12", "ZNF143"]),
            ("mouse", ["Zic5"]),
            ("drosophila", ["Cf2", "Dlip3", "Dref"]),
            ("error", [None]),
        ],
    )
    def test_compute_feature_correlation_transcription_factors(
        self,
        adata_time: AnnData,
        features: Tuple[str, List[str]],
    ):
        key_added = "test"
        rng = np.random.RandomState(42)
        adata_time = adata_time[adata_time.obs["time"].isin((0, 1))].copy()
        n0 = adata_time[adata_time.obs["time"] == 0].n_obs
        n1 = adata_time[adata_time.obs["time"] == 1].n_obs
        tmap = rng.uniform(1e-6, 1, size=(n0, n1))
        tmap /= tmap.sum().sum()
        problem = CompoundProblemWithMixin(adata_time)
        problem = problem.prepare("time", xy_callback="local-pca")
        problem[0, 1]._solution = MockSolverOutput(tmap)

        adata_time.obs[key_added] = np.hstack((np.zeros(n0), problem.pull(start=0, end=1).squeeze()))

        if features[0] == "error":
            with np.testing.assert_raises(NotImplementedError):
                res = problem.compute_feature_correlation(
                    obs_key=key_added, annotation={"celltype": ["A"]}, features=features[0]
                )
        else:
            res = problem.compute_feature_correlation(
                obs_key=key_added, annotation={"celltype": ["A"]}, features=features[0]
            )
            assert res.isnull().values.sum() == 0
            assert isinstance(res, pd.DataFrame)
            assert set(res.index) == set(features[1])

    def test_seed_reproducible(self, adata_time: AnnData):
        key_added = "test"
        rng = np.random.RandomState(42)
        adata_time = adata_time[adata_time.obs["time"].isin((0, 1))].copy()
        n0 = adata_time[adata_time.obs["time"] == 0].n_obs
        n1 = adata_time[adata_time.obs["time"] == 1].n_obs
        tmap = rng.uniform(1e-6, 1, size=(n0, n1))
        tmap /= tmap.sum().sum()
        problem = CompoundProblemWithMixin(adata_time)
        problem = problem.prepare("time", xy_callback="local-pca")
        problem[0, 1]._solution = MockSolverOutput(tmap)

        adata_time.obs[key_added] = np.hstack((np.zeros(n0), problem.pull(start=0, end=1).squeeze()))

        res_a = problem.compute_feature_correlation(obs_key=key_added, n_perms=10, n_jobs=1, seed=0, method="perm_test")
        res_b = problem.compute_feature_correlation(obs_key=key_added, n_perms=10, n_jobs=1, seed=0, method="perm_test")
        res_c = problem.compute_feature_correlation(obs_key=key_added, n_perms=10, n_jobs=1, seed=1, method="perm_test")

        assert res_a is not res_b
        np.testing.assert_array_equal(res_a.index, res_b.index)
        np.testing.assert_array_equal(res_a.columns, res_b.columns)
        np.testing.assert_allclose(res_a.values, res_b.values)

        assert not np.allclose(res_a.values, res_c.values)

    def test_seed_reproducible_parallelized(self, adata_time: AnnData):
        key_added = "test"
        rng = np.random.RandomState(42)
        adata_time = adata_time[adata_time.obs["time"].isin((0, 1))].copy()
        n0 = adata_time[adata_time.obs["time"] == 0].n_obs
        n1 = adata_time[adata_time.obs["time"] == 1].n_obs
        tmap = rng.uniform(1e-6, 1, size=(n0, n1))
        tmap /= tmap.sum().sum()
        problem = CompoundProblemWithMixin(adata_time)
        problem = problem.prepare("time", xy_callback="local-pca")
        problem[0, 1]._solution = MockSolverOutput(tmap)

        adata_time.obs[key_added] = np.hstack((np.zeros(n0), problem.pull(start=0, end=1).squeeze()))

        res_a = problem.compute_feature_correlation(
            obs_key=key_added, n_perms=10, n_jobs=2, backend="threading", seed=0, method="perm_test"
        )
        res_b = problem.compute_feature_correlation(
            obs_key=key_added, n_perms=10, n_jobs=2, backend="threading", seed=0, method="perm_test"
        )

        assert res_a is not res_b
        np.testing.assert_array_equal(res_a.index, res_b.index)
        np.testing.assert_array_equal(res_a.columns, res_b.columns)
        np.testing.assert_allclose(res_a.values, res_b.values)

    def test_confidence_level(self, adata_time: AnnData):
        key_added = "test"
        rng = np.random.RandomState(42)
        adata_time = adata_time[adata_time.obs["time"].isin((0, 1))].copy()
        n0 = adata_time[adata_time.obs["time"] == 0].n_obs
        n1 = adata_time[adata_time.obs["time"] == 1].n_obs
        tmap = rng.uniform(1e-6, 1, size=(n0, n1))
        tmap /= tmap.sum().sum()
        problem = CompoundProblemWithMixin(adata_time)
        problem = problem.prepare("time", xy_callback="local-pca")
        problem[0, 1]._solution = MockSolverOutput(tmap)

        adata_time.obs[key_added] = np.hstack((np.zeros(n0), problem.pull(start=0, end=1).squeeze()))

        res_narrow = problem.compute_feature_correlation(obs_key=key_added, confidence_level=0.95)
        res_wide = problem.compute_feature_correlation(obs_key=key_added, confidence_level=0.99)

        assert np.all(res_narrow[f"{key_added}_ci_low"] >= res_wide[f"{key_added}_ci_low"])
        assert np.all(res_narrow[f"{key_added}_ci_high"] <= res_wide[f"{key_added}_ci_high"])
