from typing import Any, Tuple, Mapping

import pytest

from ott.geometry.costs import Cosine, Euclidean, SqEuclidean

from anndata import AnnData

from moscot.problems.base import OTProblem
from moscot.solvers._output import BaseSolverOutput
from moscot.problems.generic import FGWProblem
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


class TestFGWProblem:
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

    def test_solve_balanced(self, adata_space_rotate: AnnData):  # type: ignore[no-untyped-def]
        eps = 0.5
        adata_space_rotate = adata_space_rotate[adata_space_rotate.obs["batch"].isin(("0", "1"))].copy()
        expected_keys = [("0", "1"), ("1", "2")]
        problem = FGWProblem(adata=adata_space_rotate)
        problem = problem.prepare(
            key="batch",
            policy="sequential",
            joint_attr="X_pca",
            GW_x={"attr": "obsm", "key": "spatial"},
            GW_y={"attr": "obsm", "key": "spatial"},
        )
        problem = problem.solve(epsilon=eps)

        for key, subsol in problem.solutions.items():
            assert isinstance(subsol, BaseSolverOutput)
            assert key in expected_keys

    @pytest.mark.parametrize("args_to_check", [fgw_args_1, fgw_args_2])
    def test_pass_arguments(self, adata_space_rotate: AnnData, args_to_check: Mapping[str, Any]):
        problem = FGWProblem(adata=adata_space_rotate)
        adata_space_rotate = adata_space_rotate[adata_space_rotate.obs["batch"].isin((0, 1))].copy()
        problem = problem.prepare(
            key="batch",
            policy="sequential",
            joint_attr="X_pca",
            GW_x={"attr": "obsm", "key": "spatial"},
            GW_y={"attr": "obsm", "key": "spatial"},
        )

        problem = problem.solve(**args_to_check)
        key = ("0", "1")

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

    @pytest.mark.fast()
    @pytest.mark.parametrize("cost", [("sq_euclidean", SqEuclidean), ("euclidean", Euclidean), ("cosine", Cosine)])
    def test_prepare_costs(self, adata_time: AnnData, cost: Tuple[str, Any]):
        problem = FGWProblem(adata=adata_time)
        problem = problem.prepare(
            key="time",
            policy="sequential",
            joint_attr="X_umap",
            GW_x="X_pca",
            GW_y="X_pca",
            cost=cost[0],
        )
        assert isinstance(problem[0, 1].xy.cost, cost[1])
        assert isinstance(problem[0, 1].x.cost, cost[1])
        assert isinstance(problem[0, 1].y.cost, cost[1])

    @pytest.mark.fast()
    def test_prepare_different_costs(self, adata_time: AnnData):
        problem = FGWProblem(adata=adata_time)
        problem = problem.prepare(
            key="time",
            policy="sequential",
            joint_attr="X_umap",
            GW_x="X_pca",
            GW_y="X_pca",
            cost={"xy": "cosine", "x": "euclidean", "y": "sq_euclidean"},
        )
        assert isinstance(problem[0, 1].xy.cost, Cosine)
        assert isinstance(problem[0, 1].x.cost, Euclidean)
        assert isinstance(problem[0, 1].y.cost, SqEuclidean)
