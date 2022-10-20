from typing import List

import pandas as pd
import pytest

import numpy as np

from anndata import AnnData

from moscot.problems.time import TemporalProblem
from moscot.solvers._output import BaseSolverOutput
from moscot.problems.time._lineage import BirthDeathProblem


class TestTemporalProblem:
    @pytest.mark.fast()
    def test_prepare(self, adata_time: AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = TemporalProblem(adata=adata_time)

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
        eps = 0.5
        expected_keys = [(0, 1), (1, 2)]
        problem = TemporalProblem(adata=adata_time)
        problem = problem.prepare("time")
        problem = problem.solve(epsilon=eps)

        for key, subsol in problem.solutions.items():
            assert isinstance(subsol, BaseSolverOutput)
            assert key in expected_keys

    @pytest.mark.skip(reason="Revisit this once prior and posterior marginals are implemented.")
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
        problem = TemporalProblem(adata_time)
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
        problem = TemporalProblem(adata_time)
        assert problem.proliferation_key is None

        problem.score_genes_for_marginals(gene_set_proliferation="human", gene_set_apoptosis="human")
        assert problem.proliferation_key == "proliferation"

        adata_time.obs["new_proliferation"] = np.ones(adata_time.n_obs)
        problem.proliferation_key = "new_proliferation"
        assert problem.proliferation_key == "new_proliferation"

    @pytest.mark.fast()
    def test_apoptosis_key_pipeline(self, adata_time: AnnData):
        problem = TemporalProblem(adata_time)
        assert problem.apoptosis_key is None

        problem.score_genes_for_marginals(gene_set_proliferation="human", gene_set_apoptosis="human")
        assert problem.apoptosis_key == "apoptosis"

        adata_time.obs["new_apoptosis"] = np.ones(adata_time.n_obs)
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
        problem = problem.score_genes_for_marginals(gene_set_proliferation="mouse", gene_set_apoptosis="mouse")
        problem = problem.prepare("time", a=True, b=True)
        problem = problem.solve()

        growth_rates = problem.growth_rates
        assert isinstance(growth_rates, pd.DataFrame)
        assert len(growth_rates.columns) == 1
        assert set(growth_rates.index) == set(adata_time.obs.index)
        assert set(growth_rates[growth_rates["growth_rates"].isnull()].index) == set(
            adata_time[adata_time.obs["time"] == 2].obs.index
        )
        assert set(growth_rates[~growth_rates["growth_rates"].isnull()].index) == set(
            adata_time[adata_time.obs["time"].isin([0, 1])].obs.index
        )

    def test_result_compares_to_wot(self, gt_temporal_adata: AnnData):
        # this test assures TemporalProblem returns an equivalent solution to Waddington OT (precomputed)
        adata = gt_temporal_adata.copy()
        config = gt_temporal_adata.uns
        eps = config["eps"]
        lam1 = config["lam1"]
        lam2 = config["lam2"]
        key = config["key"]
        key_1 = config["key_1"]
        key_2 = config["key_2"]
        key_3 = config["key_3"]

        tp = TemporalProblem(adata)
        tp = tp.prepare(
            key,
            subset=[(key_1, key_2), (key_2, key_3), (key_1, key_3)],
            policy="explicit",
            callback_kwargs={"joint_space": False, "n_comps": 50},
        )
        tp = tp.solve(epsilon=eps, scale_cost="mean", tau_a=lam1 / (lam1 + eps), tau_b=lam2 / (lam2 + eps))

        np.testing.assert_array_almost_equal(
            adata.uns["tmap_10_105"],
            np.array(tp[key_1, key_2].solution.transport_matrix),
        )
        np.testing.assert_array_almost_equal(
            adata.uns["tmap_105_11"],
            np.array(tp[key_2, key_3].solution.transport_matrix),
        )
        np.testing.assert_array_almost_equal(
            adata.uns["tmap_10_11"],
            np.array(tp[key_1, key_3].solution.transport_matrix),
        )

    def test_pass_arguments(self, adata_time: AnnData):
        problem = TemporalProblem(adata=adata_time)

        problem = problem.prepare(
            time_key="time",
            policy="sequential",
            filter=[(0, 1)],
        )

        args_to_check = {
            "epsilon": 0.7,
            "tau_a": 1.0,
            "tau_b": 1.0,
            "scale_cost": "max",
            "rank": 7,
            "batch_size": 123,
            "initializer": "rank_2",
            "initializer_kwargs": {},
            "jit": False,
            "threshold": 2e-3,
            "lse_mode": True,
            "norm_error": 2,
            "inner_iterations": 3,
            "min_iterations": 4,
            "max_iterations": 9,
            "gamma": 9.4,
            "gamma_rescale": False,
        }

        solver_args = {
            "lse_mode": "lse_mode",
            "threshold": "threshold",
            "norm_error": "norm_error",
            "inner_iterations": "inner_iterations",
            "min_iterations": "min_iterations",
            "max_iterations": "max_iterations",
            "initializer": "initializer",
            "initializer_kwargs": "kwargs_init",
            "jit": "jit",
        }
        lin_prob_args = {
            "epsilon": "epsilon",
            "scale_cost": "scale_cost",
            "batch_size": "batch_size",
            "tau_a": "_tau_a",
            "tau_b": "_tau_b",
        }
        geometry_args = {"epsilon": "_epsilon_init", "scale_cost": "scale_cost"}
        pointcloud_args = {
            "cost": "cost_fn",
            "power": "power",
            "batch_size": "_batch_size",
            "_scale_cost": "scale_cost",
        }

        problem = problem.solve(**args_to_check)

        solver = problem[(0, 1)]._solver._solver
        for arg in solver_args:
            assert hasattr(solver, solver_args[arg])
            assert getattr(solver, arg) == args_to_check[solver_args[arg]]

        lin_prob = problem[(0, 1)]._solver._problem
        for arg in lin_prob_args:
            assert hasattr(lin_prob, lin_prob_args[arg])
            assert getattr(lin_prob, arg) == args_to_check[lin_prob_args[arg]]

        geom = lin_prob.geom
        for arg in geometry_args:
            assert hasattr(geom, geometry_args[arg])
            assert getattr(geom, arg) == args_to_check[geometry_args[arg]]

        lin_prob.geom
        for arg in pointcloud_args:
            assert hasattr(geom, geometry_args[arg])
            assert getattr(geom, arg) == args_to_check[geometry_args[arg]]
