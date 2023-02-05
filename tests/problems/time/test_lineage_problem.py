from typing import Any, List, Mapping

import pytest

import numpy as np

from anndata import AnnData

from tests._utils import ATOL, RTOL
from moscot.problems.time import LineageProblem
from moscot.solvers._output import BaseSolverOutput
from tests.problems.conftest import (
    fgw_args_1,
    fgw_args_2,
    geometry_args,
    gw_solver_args,
    quad_prob_args,
    pointcloud_args,
    gw_linear_solver_args,
    gw_lr_linear_solver_args,
)
from moscot.problems.base._birth_death import BirthDeathProblem


class TestLineageProblem:
    @pytest.mark.fast()
    def test_prepare(self, adata_time_barcodes: AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = LineageProblem(adata=adata_time_barcodes)
        assert len(problem) == 0
        assert problem.problems == {}
        assert problem.solutions == {}
        problem = problem.prepare(
            time_key="time",
            policy="sequential",
            lineage_attr={"attr": "obsm", "key": "barcodes", "tag": "cost_matrix", "cost": "barcode_distance"},
        )

        assert isinstance(problem.problems, dict)
        assert len(problem.problems) == len(expected_keys)

        for key in problem:
            assert key in expected_keys
            assert isinstance(problem[key], BirthDeathProblem)

    def test_solve_balanced(self, adata_time_barcodes: AnnData):
        eps = 0.5
        expected_keys = [(0, 1)]
        adata_time_barcodes = adata_time_barcodes[adata_time_barcodes.obs["time"].isin((0, 1))]
        problem = LineageProblem(adata=adata_time_barcodes)
        problem = problem.prepare(
            time_key="time",
            policy="sequential",
            lineage_attr={"attr": "obsm", "key": "barcodes", "tag": "cost_matrix", "cost": "barcode_distance"},
        )
        problem = problem.solve(epsilon=eps)

        for key, subsol in problem.solutions.items():
            assert isinstance(subsol, BaseSolverOutput)
            assert key in expected_keys

    def test_solve_unbalanced(self, adata_time_barcodes: AnnData):
        taus = [9e-1, 1e-2]
        adata_time_barcodes = adata_time_barcodes[adata_time_barcodes.obs["time"].isin((0, 1))]

        problem1 = LineageProblem(adata=adata_time_barcodes)
        problem1 = problem1.prepare(
            time_key="time",
            policy="sequential",
            lineage_attr={"attr": "obsm", "key": "barcodes", "tag": "cost_matrix", "cost": "barcode_distance"},
        )
        problem2 = LineageProblem(adata=adata_time_barcodes)
        problem2 = problem2.prepare(
            time_key="time",
            policy="sequential",
            lineage_attr={"attr": "obsm", "key": "barcodes", "tag": "cost_matrix", "cost": "barcode_distance"},
        )

        assert problem1[0, 1].a is not None
        assert problem1[0, 1].b is not None
        assert problem2[0, 1].a is not None
        assert problem2[0, 1].b is not None

        problem1 = problem1.solve(epsilon=1, tau_a=taus[0], tau_b=taus[0], max_iterations=100)
        problem2 = problem2.solve(epsilon=1, tau_a=taus[1], tau_b=taus[1], max_iterations=100)

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
    def test_score_genes(self, adata_time_barcodes: AnnData, gene_set_list: List[List[str]]):
        gene_set_proliferation = gene_set_list[0]
        gene_set_apoptosis = gene_set_list[1]
        problem = LineageProblem(adata_time_barcodes)
        problem.score_genes_for_marginals(
            gene_set_proliferation=gene_set_proliferation, gene_set_apoptosis=gene_set_apoptosis
        )

        if gene_set_apoptosis is not None:
            assert problem.proliferation_key == "proliferation"
            assert adata_time_barcodes.obs["proliferation"] is not None
            assert np.sum(np.isnan(adata_time_barcodes.obs["proliferation"])) == 0
        else:
            assert problem.proliferation_key is None

        if gene_set_apoptosis is not None:
            assert problem.apoptosis_key == "apoptosis"
            assert adata_time_barcodes.obs["apoptosis"] is not None
            assert np.sum(np.isnan(adata_time_barcodes.obs["apoptosis"])) == 0
        else:
            assert problem.apoptosis_key is None

    @pytest.mark.fast()
    def test_proliferation_key_pipeline(self, adata_time_barcodes: AnnData):
        problem = LineageProblem(adata_time_barcodes)
        assert problem.proliferation_key is None

        problem.score_genes_for_marginals(gene_set_proliferation="human", gene_set_apoptosis="human")
        assert problem.proliferation_key == "proliferation"

        adata_time_barcodes.obs["new_proliferation"] = np.ones(adata_time_barcodes.n_obs)
        problem.proliferation_key = "new_proliferation"
        assert problem.proliferation_key == "new_proliferation"

    @pytest.mark.fast()
    def test_apoptosis_key_pipeline(self, adata_time_barcodes: AnnData):
        problem = LineageProblem(adata_time_barcodes)
        assert problem.apoptosis_key is None

        problem.score_genes_for_marginals(gene_set_proliferation="human", gene_set_apoptosis="human")
        assert problem.apoptosis_key == "apoptosis"

        adata_time_barcodes.obs["new_apoptosis"] = np.ones(adata_time_barcodes.n_obs)
        problem.apoptosis_key = "new_apoptosis"
        assert problem.apoptosis_key == "new_apoptosis"

    @pytest.mark.fast()
    @pytest.mark.parametrize("scaling", [0.1, 1, 4])
    def test_proliferation_key_c_pipeline(self, adata_time_barcodes: AnnData, scaling: float):
        keys = np.sort(np.unique(adata_time_barcodes.obs["time"].values))
        adata_time_barcodes = adata_time_barcodes[adata_time_barcodes.obs["time"].isin([keys[0], keys[1]])]
        delta = keys[1] - keys[0]
        problem = LineageProblem(adata_time_barcodes)
        assert problem.proliferation_key is None

        problem.score_genes_for_marginals(gene_set_proliferation="human", gene_set_apoptosis="human")
        assert problem.proliferation_key == "proliferation"

        problem = problem.prepare(
            time_key="time",
            lineage_attr={"attr": "obsm", "key": "barcodes", "tag": "cost_matrix", "cost": "barcode_distance"},
            policy="sequential",
            marginal_kwargs={"scaling": scaling},
        )
        prolif = adata_time_barcodes[adata_time_barcodes.obs["time"] == keys[0]].obs["proliferation"]
        apopt = adata_time_barcodes[adata_time_barcodes.obs["time"] == keys[0]].obs["apoptosis"]
        expected_marginals = np.exp((prolif - apopt) * delta / scaling)
        np.testing.assert_allclose(problem[keys[0], keys[1]]._prior_growth, expected_marginals, rtol=RTOL, atol=ATOL)

    @pytest.mark.fast()
    def test_barcodes_pipeline(self, adata_time_barcodes: AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = LineageProblem(adata=adata_time_barcodes)
        problem = problem.prepare(
            time_key="time",
            lineage_attr={"attr": "obsm", "key": "barcodes", "tag": "cost_matrix", "cost": "barcode_distance"},
            policy="sequential",
        )
        problem = problem.solve(max_iterations=2)

        for key in problem:
            assert key in expected_keys
            assert isinstance(problem[key], problem._base_problem_type)

    def test_custom_cost_pipeline(self, adata_time_custom_cost_xy: AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = LineageProblem(adata=adata_time_custom_cost_xy)
        problem = problem.prepare(time_key="time")
        problem = problem.solve(max_iterations=2)

        for key in problem:
            assert key in expected_keys
            assert isinstance(problem[key], BirthDeathProblem)

    def test_trees_pipeline(self, adata_time_trees: AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = LineageProblem(adata=adata_time_trees)
        problem = problem.prepare(
            time_key="time", lineage_attr={"attr": "uns", "key": "trees", "tag": "cost_matrix", "cost": "leaf_distance"}
        )
        problem = problem.solve(max_iterations=2)

        for key in problem:
            assert key in expected_keys
            assert isinstance(problem[key], BirthDeathProblem)

    def test_cell_costs_pipeline(self, adata_time_custom_cost_xy: AnnData):
        problem = LineageProblem(adata=adata_time_custom_cost_xy)
        problem = problem.prepare("time")
        problem = problem.solve(max_iterations=1)

        assert problem.cell_costs_source is None
        assert problem.cell_costs_target is None

    @pytest.mark.parametrize("args_to_check", [fgw_args_1, fgw_args_2])
    def test_pass_arguments(self, adata_time_barcodes: AnnData, args_to_check: Mapping[str, Any]):
        problem = LineageProblem(adata=adata_time_barcodes)
        adata_time_barcodes = adata_time_barcodes[adata_time_barcodes.obs["time"].isin((0, 1))]
        problem = problem.prepare(
            time_key="time",
            policy="sequential",
            lineage_attr={"attr": "obsm", "key": "barcodes", "tag": "cost_matrix", "cost": "barcode_distance"},
        )

        problem = problem.solve(**args_to_check)
        key = (0, 1)
        solver = problem[key].solver.solver
        for arg, val in gw_solver_args.items():
            assert hasattr(solver, val)
            assert getattr(solver, val) == args_to_check[arg]

        sinkhorn_solver = solver.linear_ot_solver
        lin_solver_args = gw_linear_solver_args if args_to_check["rank"] == -1 else gw_lr_linear_solver_args
        for arg, val in lin_solver_args.items():
            assert hasattr(sinkhorn_solver, val)
            el = (
                getattr(sinkhorn_solver, val)[0]
                if isinstance(getattr(sinkhorn_solver, val), tuple)
                else getattr(sinkhorn_solver, val)
            )
            args_to_c = args_to_check if arg in ["gamma", "gamma_rescale"] else args_to_check["linear_solver_kwargs"]
            assert el == args_to_c[arg]

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
