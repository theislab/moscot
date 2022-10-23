from typing import Any, Mapping

import pytest

from anndata import AnnData

from moscot.problems.time import LineageProblem
from tests.problems.conftest import (
    fgw_args_1,
    fgw_args_2,
    geometry_args,
    gw_solver_args,
    quad_prob_args,
    pointcloud_args,
    gw_sinkhorn_solver_args,
)
from moscot.problems.base._birth_death import BirthDeathProblem


class TestLineageProblem:
    @pytest.mark.fast()
    def test_barcodes_pipeline(self, adata_time_barcodes: AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = LineageProblem(adata=adata_time_barcodes)
        problem = problem.prepare(
            time_key="time",
            lineage_attr={"attr": "obsm", "key": "barcodes", "tag": "cost", "loss": "barcode_distance"},
            policy="sequential",
        )
        problem = problem.solve()

        for key in problem:
            assert key in expected_keys
            assert isinstance(problem[key], problem._base_problem_type)

    def test_custom_cost_pipeline(self, adata_time_custom_cost_xy: AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = LineageProblem(adata=adata_time_custom_cost_xy)
        problem = problem.prepare(time_key="time")
        problem = problem.solve()

        for key in problem:
            assert key in expected_keys
            assert isinstance(problem[key], BirthDeathProblem)

    @pytest.mark.skip(
        reason="Disabled passing info which cell distribution OTProblem belongs to. Hence, cannot pass trees in `uns`."
    )
    def test_trees_pipeline(self, adata_time_trees: AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = LineageProblem(adata=adata_time_trees)
        problem = problem.prepare(time_key="time", lineage_attr={"attr": "uns", "tag": "cost", "loss": "leaf_distance"})
        problem = problem.solve()

        for key in problem:
            assert key in expected_keys
            assert isinstance(problem[key], BirthDeathProblem)

    def test_cell_costs_pipeline(self, adata_time_custom_cost_xy: AnnData):
        problem = LineageProblem(adata=adata_time_custom_cost_xy)
        problem = problem.prepare("time")
        problem = problem.solve()

        assert problem.cell_costs_source is None
        assert problem.cell_costs_target is None

    @pytest.mark.parametrize("args_to_check", [fgw_args_1, fgw_args_2])
    def test_pass_arguments(self, adata_time_barcodes: AnnData, args_to_check: Mapping[str, Any]):
        problem = LineageProblem(adata=adata_time_barcodes)

        problem = problem.prepare(
            time_key="time",
            policy="sequential",
            filter=[(0, 1)],
            lineage_attr={"attr": "obsm", "key": "barcodes", "tag": "cost", "loss": "barcode_distance"},
        )

        problem = problem.solve(**args_to_check)

        solver = problem[(0, 1)]._solver._solver
        for arg in gw_solver_args:
            assert hasattr(solver, gw_solver_args[arg])
            assert getattr(solver, gw_solver_args[arg]) == args_to_check[arg]

        sinkhorn_solver = solver.linear_ot_solver
        for arg in gw_sinkhorn_solver_args:
            assert hasattr(sinkhorn_solver, gw_sinkhorn_solver_args[arg])
            assert getattr(sinkhorn_solver, gw_sinkhorn_solver_args[arg]) == args_to_check[arg]

        quad_prob = problem[(0, 1)]._solver._problem
        for arg in quad_prob_args:
            assert hasattr(quad_prob, quad_prob_args[arg])
            assert getattr(quad_prob, quad_prob_args[arg]) == args_to_check[arg]
        assert hasattr(quad_prob, "fused_penalty")
        assert quad_prob.fused_penalty == problem[(0, 1)]._solver._alpha_to_fused_penalty(args_to_check["alpha"])

        geom = quad_prob.geom_xx
        for arg in geometry_args:
            assert hasattr(geom, geometry_args[arg])
            assert getattr(geom, geometry_args[arg]) == args_to_check[arg]

        geom = quad_prob.geom_xy
        for arg in pointcloud_args:
            assert hasattr(geom, pointcloud_args[arg])
            assert getattr(geom, pointcloud_args[arg]) == args_to_check[arg]
