from typing import Any, Literal, Mapping

import pytest

import numpy as np
import pandas as pd
from ott.geometry.costs import (
    Cosine,
    ElasticL1,
    ElasticSTVS,
    Euclidean,
    PNormP,
    SqEuclidean,
    SqPNorm,
)
from ott.solvers.linear import acceleration

from anndata import AnnData

from moscot._types import CostKwargs_t
from moscot.backends.ott._utils import alpha_to_fused_penalty
from moscot.base.output import BaseSolverOutput
from moscot.base.problems import OTProblem
from moscot.problems.generic import GWProblem
from tests.problems.conftest import (
    fgw_args_1,
    fgw_args_2,
    geometry_args,
    gw_linear_solver_args,
    gw_lr_linear_solver_args,
    gw_solver_args,
    pointcloud_args,
    quad_prob_args,
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
            x_attr={"attr": "obsm", "key": "spatial"},
            y_attr={"attr": "obsm", "key": "spatial"},
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
            x_attr={"attr": "obsm", "key": "spatial"},
            y_attr={"attr": "obsm", "key": "spatial"},
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
            x_attr={"attr": "obsm", "key": "spatial"},
            y_attr={"attr": "obsm", "key": "spatial"},
        )

        problem = problem.solve(**args_to_check)
        key = ("0", "1")

        solver = problem[key].solver.solver
        for arg, val in gw_solver_args.items():
            assert getattr(solver, val, object()) == args_to_check[arg], arg

        sinkhorn_solver = solver.linear_ot_solver
        lin_solver_args = gw_linear_solver_args if args_to_check["rank"] == -1 else gw_lr_linear_solver_args
        for arg, val in lin_solver_args.items():
            el = (
                getattr(sinkhorn_solver, val)[0]
                if isinstance(getattr(sinkhorn_solver, val), tuple)
                else getattr(sinkhorn_solver, val)
            )
            assert el == args_to_check["linear_solver_kwargs"][arg], arg

        quad_prob = problem[key].solver._problem
        for arg, val in quad_prob_args.items():
            assert getattr(quad_prob, val, object()) == args_to_check[arg]
        assert quad_prob.fused_penalty == alpha_to_fused_penalty(args_to_check["alpha"])

        geom = quad_prob.geom_xx
        for arg, val in geometry_args.items():
            assert getattr(geom, val, object()) == args_to_check[arg], arg

        geom = quad_prob.geom_xy
        for arg, val in pointcloud_args.items():
            assert getattr(geom, val, object()) == args_to_check[arg], arg

    @pytest.mark.parametrize("tag", ["cost_matrix", "kernel"])
    def test_set_xy(self, adata_time: AnnData, tag: Literal["cost_matrix", "kernel"]):
        rng = np.random.RandomState(42)
        adata_time = adata_time[adata_time.obs["time"].isin((0, 1))].copy()
        problem = GWProblem(adata=adata_time)
        problem = problem.prepare(
            x_attr="X_pca",
            y_attr="X_pca",
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

    @pytest.mark.fast()
    @pytest.mark.parametrize(
        ("cost_str", "cost_inst", "cost_kwargs"),
        [
            ("sq_euclidean", SqEuclidean, {}),
            ("euclidean", Euclidean, {}),
            ("cosine", Cosine, {}),
            ("pnorm_p", PNormP, {"p": 3}),
            ("sq_pnorm", SqPNorm, {"xy": {"p": 5}, "x": {"p": 3}, "y": {"p": 4}}),
            ("elastic_l1", ElasticL1, {"gamma": 1.1}),
            ("elastic_stvs", ElasticSTVS, {"gamma": 1.2}),
        ],
    )
    def test_prepare_costs(self, adata_time: AnnData, cost_str: str, cost_inst: Any, cost_kwargs: CostKwargs_t):
        problem = GWProblem(adata=adata_time)
        problem = problem.prepare(
            key="time",
            policy="sequential",
            joint_attr="X_pca",
            alpha=0.5,
            x_attr="X_pca",
            y_attr="X_pca",
            cost=cost_str,
            cost_kwargs=cost_kwargs,
        )
        assert isinstance(problem[0, 1].x.cost, cost_inst)
        assert isinstance(problem[0, 1].y.cost, cost_inst)
        assert isinstance(problem[0, 1].xy.cost, cost_inst)

        if cost_kwargs:
            xy_items = cost_kwargs["xy"].items() if "xy" in cost_kwargs else cost_kwargs.items()
            for k, v in xy_items:
                assert getattr(problem[0, 1].xy.cost, k) == v
            x_items = cost_kwargs["x"].items() if "x" in cost_kwargs else cost_kwargs.items()
            for k, v in x_items:
                assert getattr(problem[0, 1].x.cost, k) == v
            y_items = cost_kwargs["y"].items() if "y" in cost_kwargs else cost_kwargs.items()
            for k, v in y_items:
                assert getattr(problem[0, 1].y.cost, k) == v

        problem = problem.solve(max_iterations=2)

    @pytest.mark.parametrize("tag", ["cost_matrix", "kernel"])
    def test_set_x(self, adata_time: AnnData, tag: Literal["cost_matrix", "kernel"]):
        rng = np.random.RandomState(42)
        adata_time = adata_time[adata_time.obs["time"].isin((0, 1))].copy()
        problem = GWProblem(adata=adata_time)
        problem = problem.prepare(
            x_attr="X_pca",
            y_attr="X_pca",
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
            x_attr="X_pca",
            y_attr="X_pca",
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
            x_attr="X_pca",
            y_attr="X_pca",
            cost={"xy": "cosine", "x": "euclidean", "y": "sq_euclidean"},
        )
        assert isinstance(problem[0, 1].xy.cost, Cosine)
        assert isinstance(problem[0, 1].x.cost, Euclidean)
        assert isinstance(problem[0, 1].y.cost, SqEuclidean)

    @pytest.mark.parametrize(("memory", "refresh"), [(1, 1), (5, 3), (7, 5)])
    @pytest.mark.parametrize("recenter", [True, False])
    def test_passing_ott_kwargs_linear(self, adata_space_rotate: AnnData, memory: int, refresh: int, recenter: bool):
        problem = GWProblem(adata=adata_space_rotate)
        problem = problem.prepare(
            key="batch",
            policy="sequential",
            joint_attr="X_pca",
            x_attr={"attr": "obsm", "key": "spatial"},
            y_attr={"attr": "obsm", "key": "spatial"},
        )

        problem = problem.solve(
            max_iterations=1,
            linear_solver_kwargs={
                "inner_iterations": 1,
                "max_iterations": 1,
                "anderson": acceleration.AndersonAcceleration(memory=memory, refresh_every=refresh),
                "recenter_potentials": recenter,
            },
        )

        sinkhorn_solver = problem[("0", "1")].solver.solver.linear_ot_solver

        anderson = sinkhorn_solver.anderson
        assert isinstance(anderson, acceleration.AndersonAcceleration)
        assert anderson.memory == memory
        assert anderson.refresh_every == refresh

        recenter_potentials = sinkhorn_solver.recenter_potentials
        assert recenter_potentials == recenter

    @pytest.mark.parametrize("warm_start", [True, False])
    @pytest.mark.parametrize("inner_errors", [True, False])
    def test_passing_ott_kwargs_quadratic(self, adata_space_rotate: AnnData, warm_start: bool, inner_errors: bool):
        problem = GWProblem(adata=adata_space_rotate)
        problem = problem.prepare(
            key="batch",
            policy="sequential",
            joint_attr="X_pca",
            x_attr={"attr": "obsm", "key": "spatial"},
            y_attr={"attr": "obsm", "key": "spatial"},
        )

        problem = problem.solve(max_iterations=1, warm_start=warm_start, store_inner_errors=inner_errors,
            linear_solver_kwargs={
                "inner_iterations": 1,
                "max_iterations": 1,
            })

        solver = problem[("0", "1")].solver.solver

        assert solver.warm_start == warm_start
        assert solver.store_inner_errors == inner_errors
