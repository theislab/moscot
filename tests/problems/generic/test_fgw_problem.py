from typing import Any, Mapping

import pytest

from anndata import AnnData

from moscot.problems.base import OTProblem  # type: ignore[attr-defined]
from moscot.solvers._output import BaseSolverOutput
from moscot.problems.generic import FGWProblem
from tests.problems.conftest import (
    fgw_args_1,
    fgw_args_2,
    geometry_args,
    gw_solver_args,
    quad_prob_args,
    pointcloud_args,
    gw_sinkhorn_solver_args,
)


class TestSinkhornProblem:
    @pytest.mark.fast()
    def test_prepare(self, adata_space_rotate: AnnData):
        expected_keys = [("0", "1"), ("1", "2")]
        problem = FGWProblem(adata=adata_space_rotate)

        assert len(problem) == 0
        assert problem.problems == {}
        assert problem.solutions == {}

        problem = problem.prepare(
            key="batch",
            policy="sequential",
            joint_attr="X_pca",
            GW_x={"attr": "obsm", "key": "spatial"},
            GW_y={"attr": "obsm", "key": "spatial"},
        )

        assert isinstance(problem.problems, dict)
        assert len(problem.problems) == len(expected_keys)

        for key in problem:
            assert key in expected_keys
            assert isinstance(problem[key], OTProblem)

    def test_solve_balanced(self, adata_space_rotate: AnnData):
        eps = 0.5
        expected_keys = [("0", "1"), ("1", "2")]
        problem = FGWProblem(adata=adata_space_rotate)
        problem = problem.prepare(
            key="batch",
            policy="sequential",
            joint_attr="X_pca",
            GW_x={"attr": "obsm", "key": "spatial"},
            GW_y={"attr": "obsm", "key": "spatial"},
            filter=[(0, 1)],
        )
        problem = problem.solve(epsilon=eps)

        for key, subsol in problem.solutions.items():
            assert isinstance(subsol, BaseSolverOutput)
            assert key in expected_keys

    @pytest.mark.parametrize("args_to_check", [fgw_args_1, fgw_args_2])
    def test_pass_arguments(self, adata_space_rotate: AnnData, args_to_check: Mapping[str, Any]):
        problem = FGWProblem(adata=adata_space_rotate)

        problem = problem.prepare(
            key="batch",
            policy="sequential",
            joint_attr="X_pca",
            GW_x={"attr": "obsm", "key": "spatial"},
            GW_y={"attr": "obsm", "key": "spatial"},
            filter=[(0, 1)],
        )

        problem = problem.solve(**args_to_check)
        key = ("0", "1")

        solver = problem[key]._solver._solver
        for arg in gw_solver_args:
            assert hasattr(solver, gw_solver_args[arg])
            assert getattr(solver, gw_solver_args[arg]) == args_to_check[arg]

        sinkhorn_solver = solver.linear_ot_solver
        for arg in gw_sinkhorn_solver_args:
            assert hasattr(sinkhorn_solver, gw_sinkhorn_solver_args[arg])
            assert getattr(sinkhorn_solver, gw_sinkhorn_solver_args[arg]) == args_to_check[arg]

        quad_prob = problem[key]._solver._problem
        for arg in quad_prob_args:
            assert hasattr(quad_prob, quad_prob_args[arg])
            assert getattr(quad_prob, quad_prob_args[arg]) == args_to_check[arg]
        assert hasattr(quad_prob, "fused_penalty")
        assert quad_prob.fused_penalty == problem[key]._solver._alpha_to_fused_penalty(args_to_check["alpha"])

        geom = quad_prob.geom_xx
        for arg in geometry_args:
            assert hasattr(geom, geometry_args[arg])
            assert getattr(geom, geometry_args[arg]) == args_to_check[arg]

        geom = quad_prob.geom_xy
        for arg in pointcloud_args:
            assert hasattr(geom, pointcloud_args[arg])
            assert getattr(geom, pointcloud_args[arg]) == args_to_check[arg]
