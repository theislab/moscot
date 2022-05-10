from typing import Tuple
from numbers import Number

from _utils import TestSolverOutput
import pandas as pd
import pytest

import numpy as np

from anndata import AnnData

from moscot.problems.time._lineage import TemporalProblem


class TestTemporalAnalysisMixin:
    @pytest.mark.parametrize("forward", [True, False])
    def test_cell_transition_full_pipeline(self, gt_temporal_adata: AnnData, forward: bool):
        config = gt_temporal_adata.uns["config_solution"]
        key = config["key"]
        key_1 = config["key_1"]
        key_2 = config["key_2"]
        key_3 = config["key_3"]
        cell_types = set(np.unique(gt_temporal_adata.obs["cell_type"]))
        problem = TemporalProblem(gt_temporal_adata)
        problem = problem.prepare(key, subset=[(key_1, key_2), (key_2, key_3), (key_1, key_3)], policy="explicit")
        assert set(problem.problems.keys()) == {(key_1, key_2), (key_2, key_3), (key_1, key_3)}
        problem[(key_1, key_2)]._solution = TestSolverOutput(gt_temporal_adata.uns["tmap_10_105"])
        problem[(key_2, key_3)]._solution = TestSolverOutput(gt_temporal_adata.uns["tmap_105_11"])
        problem[(key_1, key_3)]._solution = TestSolverOutput(gt_temporal_adata.uns["tmap_10_11"])

        result = problem.cell_transition(key_1, key_2, "cell_type", "cell_type", forward=forward)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (len(cell_types), len(cell_types))
        assert set(result.index) == set(cell_types)
        assert set(result.columns) == set(cell_types)
        marginal = result.sum(axis=forward == 1).values
        present_cell_type_marginal = marginal[marginal > 0]
        np.testing.assert_almost_equal(present_cell_type_marginal, np.ones(len(present_cell_type_marginal)), decimal=5)

    @pytest.mark.parametrize("forward", [True, False])
    def test_cell_transition_subset_pipeline(self, gt_temporal_adata: AnnData, forward: bool):
        config = gt_temporal_adata.uns["config_solution"]
        key = config["key"]
        key_1 = config["key_1"]
        key_2 = config["key_2"]
        key_3 = config["key_3"]
        problem = TemporalProblem(gt_temporal_adata)
        problem = problem.prepare(key, subset=[(key_1, key_2), (key_2, key_3), (key_1, key_3)], policy="explicit")
        assert set(problem.problems.keys()) == {(key_1, key_2), (key_2, key_3), (key_1, key_3)}
        problem[key_1, key_2]._solution = TestSolverOutput(gt_temporal_adata.uns["tmap_10_105"])
        problem[key_2, key_3]._solution = TestSolverOutput(gt_temporal_adata.uns["tmap_105_11"])
        problem[key_1, key_3]._solution = TestSolverOutput(gt_temporal_adata.uns["tmap_10_11"])

        early_cells = ["Stromal", "unknown"]
        late_cells = ["Stromal", "Epithelial"]
        result = problem.cell_transition(
            key_1, key_2, {"cell_type": early_cells}, {"cell_type": late_cells}, forward=forward
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)
        assert set(result.index) == set(early_cells)
        assert set(result.columns) == set(late_cells)
        marginal = result.sum(axis=forward == 1).values
        present_cell_type_marginal = marginal[marginal > 0]
        np.testing.assert_almost_equal(present_cell_type_marginal, np.ones(len(present_cell_type_marginal)), decimal=5)

    @pytest.mark.parametrize("forward", [True, False])
    def test_cell_transition_regression(self, gt_temporal_adata: AnnData, forward: bool):
        config = gt_temporal_adata.uns["config_solution"]
        key = config["key"]
        key_1 = config["key_1"]
        key_2 = config["key_2"]
        key_3 = config["key_3"]
        problem = TemporalProblem(gt_temporal_adata)
        problem = problem.prepare(key, subset=[(key_1, key_2), (key_2, key_3), (key_1, key_3)], policy="explicit")
        assert set(problem.problems.keys()) == {(key_1, key_2), (key_2, key_3), (key_1, key_3)}
        problem[key_1, key_2]._solution = TestSolverOutput(gt_temporal_adata.uns["tmap_10_105"])
        problem[key_2, key_3]._solution = TestSolverOutput(gt_temporal_adata.uns["tmap_105_11"])
        problem[key_1, key_3]._solution = TestSolverOutput(gt_temporal_adata.uns["tmap_10_11"])
        
        result = problem.cell_transition(10, 10.5, early_cells="cell_type", late_cells="cell_type", forward=forward)
        assert result.shape == (6, 6)
        marginal = result.sum(axis=forward == 1).values
        present_cell_type_marginal = marginal[marginal > 0]
        np.testing.assert_almost_equal(present_cell_type_marginal, np.ones(len(present_cell_type_marginal)), decimal=5)

        direction = "forward" if forward else "backward"
        gt = gt_temporal_adata.uns[f"cell_transition_10_105_{direction}"]
        np.testing.assert_almost_equal(result.values, gt.values, decimal=4)

    def test_compute_time_point_distances_pipeline(self, adata_time: AnnData):
        problem = TemporalProblem(adata_time)
        problem.prepare("time")
        distance_source_intermediate, distance_intermediate_target = problem.compute_time_point_distances(
            start=0, intermediate=1, end=2
        )
        assert isinstance(distance_source_intermediate, float)
        assert isinstance(distance_intermediate_target, float)
        assert distance_source_intermediate > 0
        assert distance_intermediate_target > 0

    def test_batch_distances_pipeline(self, adata_time: AnnData):
        problem = TemporalProblem(adata_time)
        problem.prepare("time")

        batch_distance = problem.compute_batch_distances(time=1, batch_key="batch")
        assert isinstance(batch_distance, float)
        assert batch_distance > 0

    @pytest.mark.parametrize("account_for_unbalancedness", [True, False])
    def test_compute_interpolated_distance_pipeline(self, gt_temporal_adata: AnnData, account_for_unbalancedness: bool):
        config = gt_temporal_adata.uns["config_solution"]
        key = config["key"]
        key_1 = config["key_1"]
        key_2 = config["key_2"]
        key_3 = config["key_3"]
        problem = TemporalProblem(gt_temporal_adata)
        problem = problem.prepare(key, subset=[(key_1, key_2), (key_2, key_3), (key_1, key_3)], policy="explicit")
        assert set(problem.problems.keys()) == {(key_1, key_2), (key_2, key_3), (key_1, key_3)}
        problem[key_1, key_2]._solution = TestSolverOutput(gt_temporal_adata.uns["tmap_10_105"])
        problem[key_2, key_3]._solution = TestSolverOutput(gt_temporal_adata.uns["tmap_105_11"])
        problem[key_1, key_3]._solution = TestSolverOutput(gt_temporal_adata.uns["tmap_10_11"])

        interpolation_result = problem.compute_interpolated_distance(
            key_1, key_2, key_3, account_for_unbalancedness=account_for_unbalancedness, seed=config["seed"]
        )
        assert isinstance(interpolation_result, float)
        assert interpolation_result > 0

    def test_compute_interpolated_distance_regression(self, gt_temporal_adata: AnnData):
        config = gt_temporal_adata.uns["config_solution"]
        key = config["key"]
        key_1 = config["key_1"]
        key_2 = config["key_2"]
        key_3 = config["key_3"]
        problem = TemporalProblem(gt_temporal_adata)
        problem = problem.prepare(key, subset=[(key_1, key_2), (key_2, key_3), (key_1, key_3)], policy="explicit")
        assert set(problem.problems.keys()) == {(key_1, key_2), (key_2, key_3), (key_1, key_3)}
        problem[key_1, key_2]._solution = TestSolverOutput(gt_temporal_adata.uns["tmap_10_105"])
        problem[key_2, key_3]._solution = TestSolverOutput(gt_temporal_adata.uns["tmap_105_11"])
        problem[key_1, key_3]._solution = TestSolverOutput(gt_temporal_adata.uns["tmap_10_11"])

        interpolation_result = problem.compute_interpolated_distance(key_1, key_2, key_3, seed=config["seed"])
        assert isinstance(interpolation_result, float)
        assert interpolation_result > 0
        np.testing.assert_almost_equal(interpolation_result, gt_temporal_adata.uns["interpolated_distance_10_105_11"], decimal=4)

    def test_compute_time_point_distances_regression(self, gt_temporal_adata: AnnData):
        config = gt_temporal_adata.uns["config_solution"]
        key = config["key"]
        key_1 = config["key_1"]
        key_2 = config["key_2"]
        key_3 = config["key_3"]
        problem = TemporalProblem(gt_temporal_adata)
        problem = problem.prepare(key, subset=[(key_1, key_2), (key_2, key_3), (key_1, key_3)], policy="explicit")
        assert set(problem.problems.keys()) == {(key_1, key_2), (key_2, key_3), (key_1, key_3)}
        problem[key_1, key_2]._solution = TestSolverOutput(gt_temporal_adata.uns["tmap_10_105"])
        problem[key_2, key_3]._solution = TestSolverOutput(gt_temporal_adata.uns["tmap_105_11"])
        problem[key_1, key_3]._solution = TestSolverOutput(gt_temporal_adata.uns["tmap_10_11"])

        result = problem.compute_time_point_distances(key_1, key_2, key_3)
        assert isinstance(result, tuple)
        assert result[0] > 0
        assert result[1] > 0
        np.testing.assert_almost_equal(result[0], gt_temporal_adata.uns["time_point_distances_10_105_11"][0], decimal=4)
        np.testing.assert_almost_equal(result[1], gt_temporal_adata.uns["time_point_distances_10_105_11"][1], decimal=4)

    def test_compute_batch_distances_regression(self, gt_temporal_adata: AnnData):
        config = gt_temporal_adata.uns["config_solution"]
        key = config["key"]
        key_1 = config["key_1"]
        key_2 = config["key_2"]
        key_3 = config["key_3"]
        problem = TemporalProblem(gt_temporal_adata)
        problem = problem.prepare(key, subset=[(key_1, key_2), (key_2, key_3), (key_1, key_3)], policy="explicit")
        assert set(problem.problems.keys()) == {(key_1, key_2), (key_2, key_3), (key_1, key_3)}
        problem[key_1, key_2]._solution = TestSolverOutput(gt_temporal_adata.uns["tmap_10_105"])
        problem[key_2, key_3]._solution = TestSolverOutput(gt_temporal_adata.uns["tmap_105_11"])
        problem[key_1, key_3]._solution = TestSolverOutput(gt_temporal_adata.uns["tmap_10_11"])

        result = problem.compute_batch_distances(key_1, "batch")
        assert isinstance(result, float)
        np.testing.assert_almost_equal(result, gt_temporal_adata.uns["batch_distances_10"], decimal=4)  # pre-computed

    def test_compute_random_distance_regression(self, gt_temporal_adata: AnnData):
        config = gt_temporal_adata.uns["config_solution"]
        key = config["key"]
        key_1 = config["key_1"]
        key_2 = config["key_2"]
        key_3 = config["key_3"]
        problem = TemporalProblem(gt_temporal_adata)
        problem = problem.prepare(key, subset=[(key_1, key_2), (key_2, key_3), (key_1, key_3)], policy="explicit")
        assert set(problem.problems.keys()) == {(key_1, key_2), (key_2, key_3), (key_1, key_3)}

        result = problem.compute_random_distance(key_1, key_2, key_3, seed=config["seed"])
        assert isinstance(result, float)
        np.testing.assert_almost_equal(result, gt_temporal_adata.uns["random_distance_10_105_11"], decimal=4)  # pre-computed

    @pytest.mark.parametrize("only_start", [True, False])
    def test_get_data_pipeline(self, adata_time: AnnData, only_start: bool):
        problem = TemporalProblem(adata_time)
        problem.prepare("time")

        result = problem._get_data(0, only_start=only_start) if only_start else problem._get_data(0, 1, 2)

        assert isinstance(result, Tuple)
        assert len(result) == 2 if only_start else len(result) == 5
        if only_start:
            assert isinstance(result[0], np.ndarray)
            assert isinstance(result[1], AnnData)
        else:
            assert isinstance(result[0], np.ndarray)
            assert isinstance(result[1], np.ndarray)
            assert isinstance(result[2], np.ndarray)
            assert isinstance(result[3], AnnData)
            assert isinstance(result[4], np.ndarray)

    @pytest.mark.parametrize("time_points", [(0, 1, 2), (0, 2, 1), ()])
    def test_get_interp_param_pipeline(self, adata_time: AnnData, time_points: Tuple[Number]):
        start, intermediate, end = time_points if len(time_points) else (42, 43, 44)
        interpolation_parameter = None if len(time_points) == 3 else 0.5
        problem = TemporalProblem(adata_time)
        problem.prepare("time")
        problem.solve()

        if intermediate <= start or end <= intermediate:
            with np.testing.assert_raises(ValueError):
                problem._get_interp_param(interpolation_parameter, start, intermediate, end)
        else:
            inter_param = problem._get_interp_param(interpolation_parameter, start, intermediate, end)
            assert inter_param == 0.5
