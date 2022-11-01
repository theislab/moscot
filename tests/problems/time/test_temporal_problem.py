from typing import Any, List, Mapping

import pandas as pd
import pytest

import numpy as np

from anndata import AnnData

from moscot.problems.time import TemporalProblem
from moscot.solvers._output import BaseSolverOutput
from tests.problems.conftest import (
    geometry_args,
    lin_prob_args,
    pointcloud_args,
    sinkhorn_args_1,
    sinkhorn_args_2,
    sinkhorn_solver_args,
)
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

    def test_solve_unbalanced(self, adata_time: AnnData):
        taus = [9e-1, 1e-2]
        problem1 = TemporalProblem(adata=adata_time)
        problem2 = TemporalProblem(adata=adata_time)
        problem1 = problem1.prepare("time", a="left_marginals", b="right_marginals")
        problem2 = problem2.prepare("time", a="left_marginals", b="right_marginals")

        assert problem1[0, 1].a is not None
        assert problem1[0, 1].b is not None
        assert problem2[0, 1].a is not None
        assert problem2[0, 1].b is not None

        problem1 = problem1.solve(epsilon=1, tau_a=taus[0], tau_b=taus[0], max_iterations=10000)
        problem2 = problem2.solve(epsilon=1, tau_a=taus[1], tau_b=taus[1], max_iterations=10000)

        assert problem1[0, 1].solution.a is not None
        assert problem1[0, 1].solution.b is not None
        assert problem2[0, 1].solution.a is not None
        assert problem2[0, 1].solution.b is not None

        div1 = np.linalg.norm(problem1[0, 1].a - problem1[0, 1].solution.a)
        div2 = np.linalg.norm(problem1[0, 1].b - problem1[0, 1].solution.b)
        assert div1 < div2

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

        growth_rates = problem.posterior_growth_rates
        assert isinstance(growth_rates, pd.DataFrame)
        assert len(growth_rates.columns) == 1
        assert set(growth_rates.index) == set(adata_time.obs.index)
        assert set(growth_rates[growth_rates["posterior_growth_rates"].isnull()].index) == set(
            adata_time[adata_time.obs["time"] == 2].obs.index
        )
        assert set(growth_rates[~growth_rates["posterior_growth_rates"].isnull()].index) == set(
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
            callback_kwargs={"n_comps": 50},
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

    @pytest.mark.parametrize("args_to_check", [sinkhorn_args_1, sinkhorn_args_2])
    def test_pass_arguments(self, adata_time: AnnData, args_to_check: Mapping[str, Any]):
        problem = TemporalProblem(adata=adata_time)
        adata_time = adata_time[adata_time.obs["time"].isin((0, 1))]

        problem = problem.prepare(
            time_key="time",
            policy="sequential",
        )

        problem = problem.solve(**args_to_check)
        key = (0, 1)
        solver = problem[key].solver.solver
        for arg, val in sinkhorn_solver_args.items():
            assert hasattr(solver, val)
            el = getattr(solver, val)[0] if isinstance(getattr(solver, val), tuple) else getattr(solver, val)
            assert el == args_to_check[arg]

        lin_prob = problem[key]._solver._problem
        for arg, val in lin_prob_args.items():
            assert hasattr(lin_prob, val)
            el = getattr(lin_prob, val)[0] if isinstance(getattr(lin_prob, val), tuple) else getattr(lin_prob, val)
            assert el == args_to_check[arg]

        geom = lin_prob.geom
        for arg, val in geometry_args.items():
            assert hasattr(geom, val)
            el = getattr(geom, val)[0] if isinstance(getattr(geom, val), tuple) else getattr(geom, val)
            assert el == args_to_check[arg]

        for arg, val in pointcloud_args.items():
            el = getattr(geom, val)[0] if isinstance(getattr(geom, val), tuple) else getattr(geom, val)
            assert hasattr(geom, val)
            if arg == "cost":
                assert type(el) == type(args_to_check[arg])  # noqa: E721
            else:
                assert el == args_to_check[arg]
