from typing import Any, Tuple, Literal, Mapping

import pandas as pd
import pytest

from ott.geometry.costs import Cosine, Euclidean, SqEuclidean
import numpy as np

from anndata import AnnData

from moscot.problems.base import OTProblem
from moscot.solvers._output import BaseSolverOutput
from moscot.problems.generic import GWProblem
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
        problem = GWProblem(adata=adata_space_rotate)

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
        adata_space_rotate = adata_space_rotate[adata_space_rotate.obs["batch"].isin(("0", "1"))].copy()
        expected_keys = [("0", "1"), ("1", "2")]
        problem = GWProblem(adata=adata_space_rotate)
        problem = problem.prepare(
            key="batch",
            policy="sequential",
            joint_attr="X_pca",
            GW_x={"attr": "obsm", "key": "spatial"},
            GW_y={"attr": "obsm", "key": "spatial"},
        )
        problem = problem.solve(alpha=0.5, epsilon=eps)

        for key, subsol in problem.solutions.items():
            assert isinstance(subsol, BaseSolverOutput)
            assert key in expected_keys

    @pytest.mark.parametrize("args_to_check", [fgw_args_1, fgw_args_2])
    def test_pass_arguments(self, adata_space_rotate: AnnData, args_to_check: Mapping[str, Any]):
        problem = GWProblem(adata=adata_space_rotate)
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
            assert getattr(solver, val, object()) == args_to_check[arg]

        sinkhorn_solver = solver.linear_ot_solver
        lin_solver_args = gw_linear_solver_args if args_to_check["rank"] == -1 else gw_lr_linear_solver_args
        for arg, val in lin_solver_args.items():
            el = (
                getattr(sinkhorn_solver, val)[0]
                if isinstance(getattr(sinkhorn_solver, val), tuple)
                else getattr(sinkhorn_solver, val)
            )
            args_to_c = args_to_check if arg in ["gamma", "gamma_rescale"] else args_to_check["linear_solver_kwargs"]
            assert el == args_to_c[arg]

        quad_prob = problem[key].solver._problem
        for arg, val in quad_prob_args.items():
            assert getattr(quad_prob, val, object()) == args_to_check[arg]
        assert quad_prob.fused_penalty == problem[key].solver._alpha_to_fused_penalty(args_to_check["alpha"])

        geom = quad_prob.geom_xx
        for arg, val in geometry_args.items():
            assert getattr(geom, val, object()) == args_to_check[arg]

        geom = quad_prob.geom_xy
        for arg, val in pointcloud_args.items():
            assert getattr(geom, val, object()) == args_to_check[arg]

    @pytest.mark.fast()
    @pytest.mark.parametrize("cost", [("sq_euclidean", SqEuclidean), ("euclidean", Euclidean), ("cosine", Cosine)])
    def test_prepare_costs(self, adata_time: AnnData, cost: Tuple[str, Any]):
        problem = GWProblem(adata=adata_time)
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

    @pytest.mark.parametrize("tag", ["cost_matrix", "kernel"])
    def test_set_xy(self, adata_time: AnnData, tag: Literal["cost_matrix", "kernel"]):
        rng = np.random.RandomState(42)
        adata_time = adata_time[adata_time.obs["time"].isin((0, 1))].copy()
        problem = GWProblem(adata=adata_time)
        problem = problem.prepare(
            GW_x="X_pca",
            GW_y="X_pca",
            joint_attr="X_pca",
            key="time",
            policy="sequential",
        )

        adata_0 = adata_time[adata_time.obs["time"] == 0]
        adata_1 = adata_time[adata_time.obs["time"] == 1]

        cm = rng.uniform(1, 10, size=(adata_0.n_obs, adata_1.n_obs))
        cost_matrix = pd.DataFrame(index=adata_0.obs_names, columns=adata_1.obs_names, data=cm)
        problem[0, 1].set_xy(cost_matrix, tag=tag)
        assert isinstance(problem[0, 1].xy.data_src, np.ndarray)
        assert problem[0, 1].xy.data_tgt is None

        # TODO(@MUCDK) once fixed in OTT-JAX test for scale_cost
        problem = problem.solve(alpha=0.5, max_iterations=5, scale_cost=1)
        assert isinstance(problem[0, 1].xy.data_src, np.ndarray)
        assert problem[0, 1].xy.data_tgt is None

    @pytest.mark.parametrize("tag", ["cost_matrix", "kernel"])
    def test_set_x(self, adata_time: AnnData, tag: Literal["cost_matrix", "kernel"]):
        rng = np.random.RandomState(42)
        adata_time = adata_time[adata_time.obs["time"].isin((0, 1))].copy()
        problem = GWProblem(adata=adata_time)
        problem = problem.prepare(
            GW_x="X_pca",
            GW_y="X_pca",
            joint_attr="X_pca",
            key="time",
            policy="sequential",
        )

        adata_0 = adata_time[adata_time.obs["time"] == 0]

        cm = rng.uniform(1, 10, size=(adata_0.n_obs, adata_0.n_obs))
        cost_matrix = pd.DataFrame(index=adata_0.obs_names, columns=adata_0.obs_names, data=cm)
        problem[0, 1].set_x(cost_matrix, tag=tag)
        assert isinstance(problem[0, 1].x.data_src, np.ndarray)
        assert problem[0, 1].x.data_tgt is None

        # TODO(@MUCDK) once fixed in OTT-JAX test for scale_cost
        problem = problem.solve(alpha=0.5, max_iterations=5, scale_cost=1)
        assert isinstance(problem[0, 1].x.data_src, np.ndarray)
        assert problem[0, 1].x.data_tgt is None

    @pytest.mark.parametrize("tag", ["cost_matrix", "kernel"])
    def test_set_y(self, adata_time: AnnData, tag: Literal["cost_matrix", "kernel"]):
        rng = np.random.RandomState(42)
        adata_time = adata_time[adata_time.obs["time"].isin((0, 1))].copy()
        problem = GWProblem(adata=adata_time)
        problem = problem.prepare(
            GW_x="X_pca",
            GW_y="X_pca",
            joint_attr="X_pca",
            key="time",
            policy="sequential",
        )

        adata_1 = adata_time[adata_time.obs["time"] == 1]

        cm = rng.uniform(1, 10, size=(adata_1.n_obs, adata_1.n_obs))
        cost_matrix = pd.DataFrame(index=adata_1.obs_names, columns=adata_1.obs_names, data=cm)
        problem[0, 1].set_y(cost_matrix, tag=tag)
        assert isinstance(problem[0, 1].y.data_src, np.ndarray)
        assert problem[0, 1].y.data_tgt is None

        # TODO(@MUCDK) once fixed in OTT-JAX test for scale_cost
        problem = problem.solve(alpha=0.5, max_iterations=5, scale_cost=1)
        assert isinstance(problem[0, 1].y.data_src, np.ndarray)
        assert problem[0, 1].y.data_tgt is None

    @pytest.mark.fast()
    def test_prepare_different_costs(self, adata_time: AnnData):
        problem = GWProblem(adata=adata_time)
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
