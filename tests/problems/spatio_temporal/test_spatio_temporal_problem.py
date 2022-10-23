from typing import Any, List, Mapping

import pytest

import numpy as np

from anndata import AnnData

from moscot.solvers._output import BaseSolverOutput
from tests.problems.conftest import (
    fgw_args_1,
    fgw_args_2,
    geometry_args,
    quad_prob_args,
    fgw_solver_args,
    pointcloud_args,
    fgw_sinkhorn_solver_args,
)
from moscot.problems.time._lineage import BirthDeathProblem
from moscot.problems.spatio_temporal import SpatioTemporalProblem


class TestSpatioTemporalProblem:
    @pytest.mark.fast()
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
        eps = 0.5
        alpha = 0.5
        expected_keys = [(0, 1), (1, 2)]
        problem = SpatioTemporalProblem(adata=adata_spatio_temporal)
        problem = problem.prepare("time", spatial_key="spatial")
        problem = problem.solve(alpha=alpha, epsilon=eps)

        for key, subsol in problem.solutions.items():
            assert isinstance(subsol, BaseSolverOutput)
            assert key in expected_keys

    @pytest.mark.skip(reason="Revisit this once prior and posterior marginals are implemented.")
    def test_solve_unbalanced(self, adata_spatio_temporal: AnnData):
        taus = [9e-1, 1e-2]
        problem1 = SpatioTemporalProblem(adata=adata_spatio_temporal)
        problem2 = SpatioTemporalProblem(adata=adata_spatio_temporal)
        problem1 = problem1.prepare("time", spatial_key="spatial", a="left_marginals", b="right_marginals")
        problem2 = problem2.prepare("time", spatial_key="spatial", a="left_marginals", b="right_marginals")
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

    @pytest.mark.parametrize("args_to_check", [fgw_args_1, fgw_args_2])
    def test_pass_arguments(self, adata_spatio_temporal: AnnData, args_to_check: Mapping[str, Any]):
        problem = SpatioTemporalProblem(adata=adata_spatio_temporal)
        problem = problem.prepare("time", spatial_key="spatial")
        problem = problem.solve(**args_to_check)

        key = (0, 1)
        solver = problem[key]._solver._solver
        for arg, val in fgw_solver_args.items():
            assert hasattr(solver, val)
            assert getattr(solver, val) == args_to_check[arg]

        sinkhorn_solver = solver.linear_ot_solver
        for arg, val in fgw_sinkhorn_solver_args.items():
            assert hasattr(sinkhorn_solver, val)
            assert getattr(sinkhorn_solver, val) == args_to_check[arg]

        quad_prob = problem[key]._solver._problem
        for arg, val in quad_prob_args.items():
            assert hasattr(quad_prob, val)
            assert getattr(quad_prob, val) == args_to_check[arg]

        assert hasattr(quad_prob, "fused_penalty")
        assert quad_prob.fused_penalty == problem[key]._solver._alpha_to_fused_penalty(args_to_check["alpha"])

        geom = quad_prob.geom_xx
        for arg, val in geometry_args.items():
            assert hasattr(geom, val)
            assert getattr(geom, val) == args_to_check[arg]

        geom = quad_prob.geom_xy
        for arg, val in pointcloud_args.items():
            assert hasattr(geom, val)
            assert getattr(geom, val) == args_to_check[arg]
