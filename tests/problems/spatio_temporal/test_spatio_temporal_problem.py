from typing import Any, List, Mapping

import pytest

import numpy as np
import pandas as pd

from anndata import AnnData

from moscot.backends.ott._utils import alpha_to_fused_penalty
from moscot.base.output import BaseDiscreteSolverOutput
from moscot.base.problems import BirthDeathProblem
from moscot.problems.spatiotemporal import SpatioTemporalProblem
from tests._utils import ATOL, RTOL
from tests.problems.conftest import (
    fgw_args_1,
    fgw_args_2,
    geometry_args,
    gw_linear_solver_args,
    gw_lr_linear_solver_args,
    gw_lr_solver_args,
    gw_solver_args,
    pointcloud_args,
    quad_prob_args,
)


class TestSpatioTemporalProblem:
    @pytest.mark.fast
    def test_prepare(self, adata_spatio_temporal: AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = SpatioTemporalProblem(adata=adata_spatio_temporal)

        assert len(problem) == 0
        assert problem.problems == {}
        assert problem.solutions == {}

        problem = problem.prepare(
            time_key="time",
            spatial_key="spatial",
        )

        assert isinstance(problem.problems, dict)
        assert len(problem.problems) == len(expected_keys)

        for key in problem:
            assert key in expected_keys
            assert isinstance(problem[key], BirthDeathProblem)

    def test_solve_balanced(self, adata_spatio_temporal: AnnData):
        eps = 1
        alpha = 0.5
        expected_keys = [(0, 1), (1, 2)]
        problem = SpatioTemporalProblem(adata=adata_spatio_temporal)
        problem = problem.prepare("time", spatial_key="spatial")
        problem = problem.solve(alpha=alpha, epsilon=eps)

        for key, subsol in problem.solutions.items():
            assert isinstance(subsol, BaseDiscreteSolverOutput)
            assert key in expected_keys

    @pytest.mark.skip(reason="unbalanced does not work yet: https://github.com/ott-jax/ott/issues/519")
    def test_solve_unbalanced(self, adata_spatio_temporal: AnnData):
        taus = [9e-1, 1e-2]
        problem1 = SpatioTemporalProblem(adata=adata_spatio_temporal)
        problem2 = SpatioTemporalProblem(adata=adata_spatio_temporal)
        problem1 = problem1.prepare(
            "time", spatial_key="spatial", a="left_marginals_unbalanced", b="right_marginals_unbalanced"
        )
        problem2 = problem2.prepare(
            "time", spatial_key="spatial", a="left_marginals_unbalanced", b="right_marginals_unbalanced"
        )
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

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "gene_set_list",
        [
            [None, None],
            ["human", "human"],
            ["mouse", "mouse"],
            [["ANLN", "ANP32E", "ATAD2"], ["ADD1", "AIFM3", "ANKH"]],
        ],
    )
    def test_score_genes(self, adata_spatio_temporal: AnnData, gene_set_list: List[List[str]]):
        gene_set_proliferation = gene_set_list[0]
        gene_set_apoptosis = gene_set_list[1]
        problem = SpatioTemporalProblem(adata_spatio_temporal)
        problem.score_genes_for_marginals(
            gene_set_proliferation=gene_set_proliferation, gene_set_apoptosis=gene_set_apoptosis
        )

        if gene_set_apoptosis is not None:
            assert problem.proliferation_key == "proliferation"
            assert adata_spatio_temporal.obs["proliferation"] is not None
            assert np.sum(np.isnan(adata_spatio_temporal.obs["proliferation"])) == 0
        else:
            assert problem.proliferation_key is None

        if gene_set_apoptosis is not None:
            assert problem.apoptosis_key == "apoptosis"
            assert adata_spatio_temporal.obs["apoptosis"] is not None
            assert np.sum(np.isnan(adata_spatio_temporal.obs["apoptosis"])) == 0
        else:
            assert problem.apoptosis_key is None

    @pytest.mark.fast
    def test_proliferation_key_pipeline(self, adata_spatio_temporal: AnnData):
        problem = SpatioTemporalProblem(adata_spatio_temporal)
        assert problem.proliferation_key is None

        problem.score_genes_for_marginals(gene_set_proliferation="human", gene_set_apoptosis="human")
        assert problem.proliferation_key == "proliferation"

        adata_spatio_temporal.obs["new_proliferation"] = np.ones(adata_spatio_temporal.n_obs)
        problem.proliferation_key = "new_proliferation"
        assert problem.proliferation_key == "new_proliferation"

    @pytest.mark.fast
    def test_apoptosis_key_pipeline(self, adata_spatio_temporal: AnnData):
        problem = SpatioTemporalProblem(adata_spatio_temporal)
        assert problem.apoptosis_key is None

        problem.score_genes_for_marginals(gene_set_proliferation="human", gene_set_apoptosis="human")
        assert problem.apoptosis_key == "apoptosis"

        adata_spatio_temporal.obs["new_apoptosis"] = np.ones(adata_spatio_temporal.n_obs)
        problem.apoptosis_key = "new_apoptosis"
        assert problem.apoptosis_key == "new_apoptosis"

    @pytest.mark.fast
    @pytest.mark.parametrize("scaling", [0.1, 1, 4])
    def test_proliferation_key_c_pipeline(self, adata_spatio_temporal: AnnData, scaling: float):
        key0, key1, *_ = np.sort(np.unique(adata_spatio_temporal.obs["time"].values))
        adata_spatio_temporal = adata_spatio_temporal[adata_spatio_temporal.obs["time"].isin([key0, key1])].copy()
        delta = key1 - key0
        problem = SpatioTemporalProblem(adata_spatio_temporal)
        assert problem.proliferation_key is None

        problem.score_genes_for_marginals(gene_set_proliferation="human", gene_set_apoptosis="human")
        assert problem.proliferation_key == "proliferation"

        problem = problem.prepare(time_key="time", marginal_kwargs={"scaling": scaling})
        prolif = adata_spatio_temporal[adata_spatio_temporal.obs["time"] == key0].obs["proliferation"]
        apopt = adata_spatio_temporal[adata_spatio_temporal.obs["time"] == key0].obs["apoptosis"]
        expected_marginals = np.exp((prolif - apopt) * delta / scaling)
        np.testing.assert_allclose(problem[key0, key1]._prior_growth, expected_marginals, rtol=RTOL, atol=ATOL)

    def test_growth_rates_pipeline(self, adata_spatio_temporal: AnnData):
        problem = SpatioTemporalProblem(adata=adata_spatio_temporal)
        problem = problem.score_genes_for_marginals(gene_set_proliferation="mouse", gene_set_apoptosis="mouse")
        problem = problem.prepare("time", a=True, b=True)
        problem = problem.solve(max_iterations=2)

        growth_rates = problem.posterior_growth_rates
        assert isinstance(growth_rates, pd.DataFrame)
        assert len(growth_rates.columns) == 1
        assert set(growth_rates.index) == set(adata_spatio_temporal.obs.index)
        assert set(growth_rates[growth_rates["posterior_growth_rates"].isnull()].index) == set(
            adata_spatio_temporal[adata_spatio_temporal.obs["time"] == 2].obs.index
        )
        assert set(growth_rates[~growth_rates["posterior_growth_rates"].isnull()].index) == set(
            adata_spatio_temporal[adata_spatio_temporal.obs["time"].isin([0, 1])].obs.index
        )

    def test_cell_costs_pipeline(self, adata_spatio_temporal: AnnData):
        problem = SpatioTemporalProblem(adata=adata_spatio_temporal)
        problem = problem.prepare("time")
        problem = problem.solve(max_iterations=2)

        assert problem.cell_costs_source is None
        assert problem.cell_costs_target is None

    @pytest.mark.parametrize("args_to_check", [fgw_args_1, fgw_args_2])
    def test_pass_arguments(self, adata_spatio_temporal: AnnData, args_to_check: Mapping[str, Any]):
        problem = SpatioTemporalProblem(adata=adata_spatio_temporal)
        problem = problem.prepare("time", spatial_key="spatial")
        problem = problem.solve(**args_to_check)

        key = (0, 1)
        solver = problem[key].solver.solver
        args = gw_solver_args if args_to_check["rank"] == -1 else gw_lr_solver_args
        for arg, val in args.items():
            assert hasattr(solver, val)
            if arg != "initializer":
                assert getattr(solver, val) == args_to_check[arg], arg

        sinkhorn_solver = solver.linear_solver if args_to_check["rank"] == -1 else solver
        lin_solver_args = gw_linear_solver_args if args_to_check["rank"] == -1 else gw_lr_linear_solver_args
        tmp_dict = args_to_check["linear_solver_kwargs"] if args_to_check["rank"] == -1 else args_to_check
        for arg, val in lin_solver_args.items():
            el = (
                getattr(sinkhorn_solver, val)[0]
                if isinstance(getattr(sinkhorn_solver, val), tuple)
                else getattr(sinkhorn_solver, val)
            )
            assert el == tmp_dict[arg], arg

        quad_prob = problem[key]._solver._problem
        for arg, val in quad_prob_args.items():
            assert hasattr(quad_prob, val)
            assert getattr(quad_prob, val) == args_to_check[arg]
        assert hasattr(quad_prob, "fused_penalty")
        assert quad_prob.fused_penalty == alpha_to_fused_penalty(args_to_check["alpha"])

        geom = quad_prob.geom_xx
        for arg, val in geometry_args.items():
            assert hasattr(geom, val)
            el = getattr(geom, val)[0] if isinstance(getattr(geom, val), tuple) else getattr(geom, val)
            if arg == "epsilon":
                eps_processed = getattr(geom, val)
                assert eps_processed == args_to_check[arg], arg
            else:
                assert getattr(geom, val) == args_to_check[arg], arg
                assert el == args_to_check[arg]

        geom = quad_prob.geom_xy
        for arg, val in pointcloud_args.items():
            assert hasattr(geom, val)
            assert getattr(geom, val) == args_to_check[arg]
