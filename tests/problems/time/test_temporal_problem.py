from typing import List

import pandas as pd
import pytest

import numpy as np

from anndata import AnnData

from moscot.solvers._output import BaseSolverOutput
from moscot.problems.time._lineage import TemporalProblem, TemporalBaseProblem


class TestTemporalProblem:
    def test_prepare(self, adata_time: AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = TemporalProblem(adata=adata_time)

        assert len(problem) == 0
        assert problem.problems is None
        assert problem.solutions is None

        problem = problem.prepare(
            time_key="time",
            axis="obs",
            policy="sequential",
        )

        for key in problem:
            assert key in expected_keys
            assert isinstance(problem[key], TemporalBaseProblem)

    def test_solve_balanced(self, adata_time: AnnData):
        eps = 0.5
        expected_keys = [(0, 1), (1, 2)]
        problem = TemporalProblem(adata=adata_time)
        problem = problem.prepare("time")
        problem = problem.solve(epsilon=eps)

        for key, subsol in problem.solutions.items():
            assert isinstance(subsol, BaseSolverOutput)
            assert key in expected_keys

    def test_solve_unbalanced(self, adata_time: AnnData):
        taus = [9e-1, 1e-2]
        problem1 = TemporalProblem(adata=adata_time)
        problem2 = TemporalProblem(adata=adata_time)
        problem1 = problem1.prepare("time", a="left_marginals", b="right_marginals")
        problem2 = problem2.prepare("time", a="left_marginals", b="right_marginals")
        problem1 = problem1.solve(tau_a=taus[0], tau_b=taus[0])
        problem2 = problem2.solve(tau_a=taus[1], tau_b=taus[1])

        assert problem1[0, 1].a is not None
        assert problem1[0, 1].b is not None
        assert problem2[0, 1].a is not None
        assert problem2[0, 1].b is not None

        div1 = np.linalg.norm(problem1[0, 1].a[:, -1] - np.ones(len(problem1[0, 1].a[:, -1])))
        div2 = np.linalg.norm(problem2[0, 1].a[:, -1] - np.ones(len(problem2[0, 1].a[:, -1])))
        assert div1 <= div2

    @pytest.mark.parametrize(
        "n_iters", [3]
    )  # TODO(@MUCDK) as soon as @michalk8 unified warnings/errors test for negative value
    def test_multiple_iterations(self, adata_time: AnnData, n_iters: int):
        problem = TemporalProblem(adata=adata_time)
        problem = problem.prepare("time")
        problem = problem.solve(n_iters=n_iters)

        assert problem[0, 1].growth_rates.shape[1] == n_iters + 1
        assert np.all(
            problem[0, 1].growth_rates[:, 0] == np.ones(len(problem[0, 1].a[:, -1])) / len(problem[0, 1].a[:, -1])
        )
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_almost_equal,
            problem[0, 1].growth_rates[:, 0],
            problem[0, 1].growth_rates[:, 1],
        )

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
        problem = TemporalProblem(adata_time)
        problem = problem.score_genes_for_marginals(
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

    def test_proliferation_key_pipeline(self, adata_time: AnnData):
        problem = TemporalProblem(adata_time)

        assert problem.proliferation_key is None

        problem.score_genes_for_marginals(gene_set_proliferation="human", gene_set_apoptosis="human")

        assert problem.proliferation_key == "proliferation"

        problem.proliferation_key = "new_proliferation"

        assert problem.proliferation_key == "new_proliferation"

    def test_apoptosis_key_pipeline(self, adata_time: AnnData):
        problem = TemporalProblem(adata_time)

        assert problem.apoptosis_key is None

        problem.score_genes_for_marginals(gene_set_proliferation="human", gene_set_apoptosis="human")

        assert problem.apoptosis_key == "apoptosis"

        problem.apoptosis_key = "new_apoptosis"

        assert problem.apoptosis_key == "new_apoptosis"

    def test_cell_costs_source_pipeline(self, adata_time: AnnData):
        problem = TemporalProblem(adata=adata_time)
        problem = problem.prepare("time")
        problem = problem.solve()

        cell_costs_source = problem.cell_costs_source

        assert isinstance(cell_costs_source, pd.DataFrame)
        assert len(cell_costs_source.columns) == 1
        assert list(cell_costs_source.columns)[0] == "cell_cost_source"
        assert set(cell_costs_source.index) == set(adata_time.obs.index)
        assert set(cell_costs_source[cell_costs_source["cell_cost_source"].isnull()].index) == set(
            adata_time[adata_time.obs["time"] == 2].obs.index
        )
        assert set(cell_costs_source[~cell_costs_source["cell_cost_source"].isnull()].index) == set(
            adata_time[adata_time.obs["time"].isin([0, 1])].obs.index
        )

    def test_cell_costs_target_pipeline(self, adata_time: AnnData):
        problem = TemporalProblem(adata=adata_time)
        problem = problem.prepare("time")
        problem = problem.solve()

        cell_costs_target = problem.cell_costs_target

        assert isinstance(cell_costs_target, pd.DataFrame)
        assert len(cell_costs_target.columns) == 1
        assert list(cell_costs_target.columns)[0] == "cell_cost_target"
        assert set(cell_costs_target.index) == set(adata_time.obs.index)
        assert set(cell_costs_target[cell_costs_target["cell_cost_target"].isnull()].index) == set(
            adata_time[adata_time.obs["time"] == 0].obs.index
        )
        assert set(cell_costs_target[~cell_costs_target["cell_cost_target"].isnull()].index) == set(
            adata_time[adata_time.obs["time"].isin([1, 2])].obs.index
        )

    def test_growth_rates_pipeline(self, adata_time: AnnData):
        problem = TemporalProblem(adata=adata_time)
        problem = problem.prepare("time")
        problem = problem.solve()

        growth_rates = problem.growth_rates
        assert isinstance(growth_rates, pd.DataFrame)
        assert len(growth_rates.columns) == 2
        assert set(growth_rates.index) == set(adata_time.obs.index)
        assert set(growth_rates[growth_rates["g_0"].isnull()].index) == set(
            growth_rates[growth_rates["g_1"].isnull()].index
        )
        assert set(growth_rates[growth_rates["g_0"].isnull()].index) == set(
            adata_time[adata_time.obs["time"] == 2].obs.index
        )
        assert set(growth_rates[~growth_rates["g_0"].isnull()].index) == set(
            adata_time[adata_time.obs["time"].isin([0, 1])].obs.index
        )
