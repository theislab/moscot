from typing import List

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from moscot.problems.time import TemporalNeuralProblem
from moscot.base.output import BaseSolverOutput
from moscot.problems.time._lineage import BirthDeathProblem
from tests._utils import ATOL, RTOL
from tests.problems.conftest import (
    neuraldual_args_1,
    neuraldual_args_2,
    neuraldual_solver_args
)


class TestTemporalNeuralProblem:

    @pytest.mark.fast()
    def test_prepare(self, adata_time: ad.AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = TemporalNeuralProblem(adata=adata_time)

        assert len(problem) == 0
        assert problem.problems == {}
        assert problem.solutions == {}

        problem = problem.prepare(time_key="time", joint_attr="X_pca", policy="sequential")

        assert isinstance(problem.problems, dict)
        assert len(problem.problems) == len(expected_keys)

        for key in problem:
            assert key in expected_keys
            assert isinstance(problem[key], BirthDeathProblem)

    def test_solve_balanced_no_baseline(self, adata_time: ad.AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = TemporalNeuralProblem(adata=adata_time)
        problem = problem.prepare(time_key="time", joint_attr="X_pca")
        problem = problem.solve(**neuraldual_args_1)

        for key, subsol in problem.solutions.items():
            assert isinstance(subsol, BaseSolverOutput)
            assert key in expected_keys

    def test_solve_unbalanced_with_baseline(self, adata_time: ad.AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = TemporalNeuralProblem(adata=adata_time)
        problem = problem.prepare(time_key="time", joint_attr="X_pca")
        problem = problem.solve(**neuraldual_args_2)

        for key, subsol in problem.solutions.items():
            assert isinstance(subsol, BaseSolverOutput)
            assert key in expected_keys

    def test_reproducibility(self, adata_time: ad.AnnData):
        pc_tzero = adata_time[adata_time.obs["time"] == 0].obsm["X_pca"]
        problem_one = TemporalNeuralProblem(adata=adata_time)
        problem_one = problem_one.prepare(time_key="time", joint_attr="X_pca")
        problem_one = problem_one.solve(**neuraldual_args_1)

        problem_two = TemporalNeuralProblem(adata=adata_time)
        problem_two = problem_one.prepare("time", joint_attr="X_pca")
        problem_two = problem_one.solve(**neuraldual_args_1)

        for key in problem_one.solutions.keys():
            assert np.allclose(
                problem_one[key].solution.push(pc_tzero),
                problem_two[key].solution.push(pc_tzero),
                rtol=RTOL,
                atol=ATOL,
            )
            assert np.allclose(
                problem_one[key].solution.pull(pc_tzero),
                problem_two[key].solution.pull(pc_tzero),
                rtol=RTOL,
                atol=ATOL,
            )

    @pytest.mark.fast()
    @pytest.mark.parametrize(
        "gene_set_list",
        [
            [None, None],
            ["human", "human"],
            ["mouse", "mouse"],
            [["ANLN", "ANP32E", "ATAD2"], ["ADD1", "AIFM3", "ANKH"]],
        ],
    )
    def test_score_genes(self, adata_time: ad.AnnData, gene_set_list: List[List[str]]):
        gene_set_proliferation = gene_set_list[0]
        gene_set_apoptosis = gene_set_list[1]
        problem = TemporalNeuralProblem(adata_time)
        problem.score_genes_for_marginals(
            gene_set_proliferation=gene_set_proliferation, gene_set_apoptosis=gene_set_apoptosis
        )

        if gene_set_apoptosis is not None:
            assert problem.proliferation_key == "proliferation"
            assert adata_time.obs["proliferation"] is not None
            assert np.sum(np.isnan(adata_time.obs["proliferation"])) == 0
        else:
            assert problem.proliferation_key is None

        if gene_set_apoptosis is not None:
            assert problem.apoptosis_key == "apoptosis"
            assert adata_time.obs["apoptosis"] is not None
            assert np.sum(np.isnan(adata_time.obs["apoptosis"])) == 0
        else:
            assert problem.apoptosis_key is None

    @pytest.mark.fast()
    def test_proliferation_key_pipeline(self, adata_time: ad.AnnData):
        problem = TemporalNeuralProblem(adata_time)
        assert problem.proliferation_key is None

        problem.score_genes_for_marginals(gene_set_proliferation="human", gene_set_apoptosis="human")
        assert problem.proliferation_key == "proliferation"

        adata_time.obs["new_proliferation"] = np.ones(adata_time.n_obs)
        problem.proliferation_key = "new_proliferation"
        assert problem.proliferation_key == "new_proliferation"

    @pytest.mark.fast()
    def test_apoptosis_key_pipeline(self, adata_time: ad.AnnData):
        problem = TemporalNeuralProblem(adata_time)
        assert problem.apoptosis_key is None

        problem.score_genes_for_marginals(gene_set_proliferation="human", gene_set_apoptosis="human")
        assert problem.apoptosis_key == "apoptosis"

        adata_time.obs["new_apoptosis"] = np.ones(adata_time.n_obs)
        problem.apoptosis_key = "new_apoptosis"
        assert problem.apoptosis_key == "new_apoptosis"

    @pytest.mark.fast()
    @pytest.mark.parametrize("scaling", [0.1, 1, 4])
    def test_proliferation_key_c_pipeline(self, adata_time: ad.AnnData, scaling: float):
        keys = np.sort(np.unique(adata_time.obs["time"].values))
        adata_time = adata_time[adata_time.obs["time"].isin([keys[0], keys[1]])]
        delta = keys[1] - keys[0]
        problem = TemporalNeuralProblem(adata_time)
        assert problem.proliferation_key is None

        problem.score_genes_for_marginals(gene_set_proliferation="human", gene_set_apoptosis="human")
        assert problem.proliferation_key == "proliferation"

        problem = problem.prepare(time_key="time", marginal_kwargs={"scaling": scaling})
        prolif = adata_time[adata_time.obs["time"] == keys[0]].obs["proliferation"]
        apopt = adata_time[adata_time.obs["time"] == keys[0]].obs["apoptosis"]
        expected_marginals = np.exp((prolif - apopt) * delta / scaling)
        np.testing.assert_allclose(problem[keys[0], keys[1]]._prior_growth, expected_marginals, rtol=RTOL, atol=ATOL)

    def test_pass_arguments(self, adata_time: ad.AnnData):
        problem = TemporalNeuralProblem(adata=adata_time)
        adata_time = adata_time[adata_time.obs["time"].isin((0, 1))]
        problem = problem.prepare(time_key="time", joint_attr="X_pca")
        problem = problem.solve(**neuraldual_args_1)

        key = (0, 1)
        solver = problem[key].solver.solver
        assert solver.conditional == False
        print("solver is ", solver)
        print("neuraldual_solver_args", neuraldual_solver_args)
        for arg, val in neuraldual_solver_args.items():
            assert hasattr(solver, val)
            el = getattr(solver, val)[0] if isinstance(getattr(solver, val), tuple) else getattr(solver, val)
            assert el == neuraldual_args_1[arg]
    
    @pytest.mark.parametrize("forward", [True, False])
    def test_cell_transition_full_pipeline(self, gt_temporal_adata: ad.AnnData, forward: bool):
        config = gt_temporal_adata.uns
        key = config["key"]
        key_1 = config["key_1"]
        key_2 = config["key_2"]
        key_3 = config["key_3"]
        cell_types = set(gt_temporal_adata.obs["cell_type"].cat.categories)
        problem = TemporalNeuralProblem(gt_temporal_adata)
        problem = problem.prepare(key)
        assert set(problem.problems.keys()) == {(key_1, key_2), (key_2, key_3)}
        problem = problem.solve(**neuraldual_args_1)

        cell_types_present_key_1 = (
            gt_temporal_adata[gt_temporal_adata.obs[key] == key_1].obs["cell_type"].cat.categories
        )
        cell_types_present_key_2 = (
            gt_temporal_adata[gt_temporal_adata.obs[key] == key_2].obs["cell_type"].cat.categories
        )

        result = problem.cell_transition(
            key_1,
            key_2,
            "cell_type",
            "cell_type",
            forward=forward,
        )
        assert isinstance(result, pd.DataFrame)
        expected_shape = (len(cell_types_present_key_1), len(cell_types_present_key_2))
        assert result.shape == expected_shape
        assert set(result.index) == set(cell_types_present_key_1) if forward else set(cell_types)
        assert set(result.columns) == set(cell_types_present_key_2) if not forward else set(cell_types)
        marginal = result.sum(axis=forward == 1).values
        present_cell_type_marginal = marginal[marginal > 0]
        np.testing.assert_almost_equal(present_cell_type_marginal, 1, decimal=5)

    @pytest.mark.fast()
    @pytest.mark.parametrize("forward", [True, False])
    def test_cell_transition_subset_pipeline(self, gt_temporal_adata: ad.AnnData, forward: bool):
        config = gt_temporal_adata.uns
        key = config["key"]
        key_1 = config["key_1"]
        key_2 = config["key_2"]
        key_3 = config["key_3"]
        problem = TemporalNeuralProblem(gt_temporal_adata)
        problem = problem.prepare(key)
        assert set(problem.problems.keys()) == {(key_1, key_2), (key_2, key_3)}
        problem = problem.solve(**neuraldual_args_1)

        early_annotation = ["Stromal", "unknown"]# if forward else ["Stromal", "Epithelial"]
        late_annotation = ["Stromal", "Epithelial"]# if forward else ["Stromal", "unknown"]
        result = problem.cell_transition(
            key_1,
            key_2,
            {"cell_type": early_annotation},
            {"cell_type": late_annotation},
            forward=forward,
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (len(early_annotation), len(late_annotation))
        assert set(result.index) == set(early_annotation)
        assert set(result.columns) == set(late_annotation)

        marginal = result.sum(axis=forward == 1).values
        present_cell_type_marginal = marginal[marginal > 0]
        np.testing.assert_almost_equal(present_cell_type_marginal, np.ones(len(present_cell_type_marginal)), decimal=5)