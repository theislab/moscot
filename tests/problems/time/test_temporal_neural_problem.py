from moscot.datasets import simulate_data
from typing import Any, List, Mapping

import pandas as pd
import pytest

import numpy as np

from anndata import AnnData

from tests._utils import ATOL, RTOL
from moscot.problems.time import TemporalNeuralProblem
from moscot.solvers._output import BaseSolverOutput
from tests.problems.conftest import (
    neuraldual_args_1,
    neuraldual_args_2,
)
from moscot.problems.time._lineage import BirthDeathProblem


class TestTemporalNeuralProblem:
    
    @pytest.mark.fast()
    def test_prepare(self, adata_time: AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = TemporalNeuralProblem(adata=adata_time)

        assert len(problem) == 0
        assert problem.problems == {}
        assert problem.solutions == {}

        problem = problem.prepare(
            time_key="time",
            policy="sequential",
        )

        assert isinstance(problem.problems, dict)
        assert len(problem.problems) == len(expected_keys)

        for key in problem:
            assert key in expected_keys
            assert isinstance(problem[key], BirthDeathProblem)

    def test_solve_balanced(self, adata_time: AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = TemporalNeuralProblem(adata=adata_time)
        problem = problem.prepare("time")
        problem = problem.solve(**neuraldual_args_1)

        for key, subsol in problem.solutions.items():
            assert isinstance(subsol, BaseSolverOutput)
            assert key in expected_keys

    def test_solve_unbalanced(self, adata_time: AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = TemporalNeuralProblem(adata=adata_time)
        problem = problem.prepare("time")
        problem = problem.solve(**neuraldual_args_2)

        for key, subsol in problem.solutions.items():
            assert isinstance(subsol, BaseSolverOutput)
            assert key in expected_keys

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
    def test_score_genes(self, adata_time: AnnData, gene_set_list: List[List[str]]):
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
    def test_proliferation_key_pipeline(self, adata_time: AnnData):
        problem = TemporalNeuralProblem(adata_time)
        assert problem.proliferation_key is None

        problem.score_genes_for_marginals(gene_set_proliferation="human", gene_set_apoptosis="human")
        assert problem.proliferation_key == "proliferation"

        adata_time.obs["new_proliferation"] = np.ones(adata_time.n_obs)
        problem.proliferation_key = "new_proliferation"
        assert problem.proliferation_key == "new_proliferation"

    @pytest.mark.fast()
    def test_apoptosis_key_pipeline(self, adata_time: AnnData):
        problem = TemporalNeuralProblem(adata_time)
        assert problem.apoptosis_key is None

        problem.score_genes_for_marginals(gene_set_proliferation="human", gene_set_apoptosis="human")
        assert problem.apoptosis_key == "apoptosis"

        adata_time.obs["new_apoptosis"] = np.ones(adata_time.n_obs)
        problem.apoptosis_key = "new_apoptosis"
        assert problem.apoptosis_key == "new_apoptosis"

    @pytest.mark.fast()
    @pytest.mark.parametrize("scaling", [0.1, 1, 4])
    def test_proliferation_key_c_pipeline(self, adata_time: AnnData, scaling: float):
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
