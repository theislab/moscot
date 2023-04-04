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

from anndata import AnnData

from moscot.base.output import BaseSolverOutput
from moscot.base.problems import OTProblem
from moscot.problems.generic import GWProblem
from tests.problems.conftest import (
    geometry_args,
    gw_args_1,
    gw_args_2,
    gw_linear_solver_args,
    gw_lr_linear_solver_args,
    gw_solver_args,
    quad_prob_args,
)


class TestGWProblem:
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
            x_attr={"attr": "obsm", "key": "spatial"},
            y_attr={"attr": "obsm", "key": "spatial"},
        )

        assert isinstance(problem.problems, dict)
        assert len(problem.problems) == len(expected_keys)

        for key in problem:
            assert key in expected_keys
            assert isinstance(problem[key], OTProblem)

    def test_solve_balanced(self, adata_space_rotate: AnnData):  # type: ignore[no-untyped-def]
        eps = 0.5
        expected_keys = [("0", "1"), ("1", "2")]
        problem = GWProblem(adata=adata_space_rotate)
        problem = problem.prepare(
            key="batch",
            policy="sequential",
            x_attr={"attr": "obsm", "key": "spatial"},
            y_attr={"attr": "obsm", "key": "spatial"},
        )
        problem = problem.solve(epsilon=eps)

        for key, subsol in problem.solutions.items():
            assert isinstance(subsol, BaseSolverOutput)
            assert key in expected_keys
            assert problem[key].solver._problem.geom_xy is None

    @pytest.mark.parametrize("method", ["fischer", "perm_test"])
    def test_compute_feature_correlation(self, adata_space_rotate: AnnData, method: str):
        problem = GWProblem(adata=adata_space_rotate)
        problem = problem.prepare(
            key="batch",
            policy="sequential",
            x_attr={"attr": "obsm", "key": "spatial"},
            y_attr={"attr": "obsm", "key": "spatial"},
        )
        problem = problem.solve(epsilon=0.5)
        assert problem["0", "1"].solution.converged

        key_added = "test_push"
        problem.push(source="0", target="1", data="celltype", subset="A", key_added=key_added)
        feature_correlation = problem.compute_feature_correlation(key_added, significance_method=method)

        assert isinstance(feature_correlation, pd.DataFrame)
        suffix = ["_corr", "_pval", "_qval", "_ci_low", "_ci_high"]
        assert list(feature_correlation.columns) == [key_added + suf for suf in suffix]
        assert feature_correlation.isna().sum().sum() == 0

    @pytest.mark.parametrize("args_to_check", [gw_args_1, gw_args_2])
    def test_pass_arguments(self, adata_space_rotate: AnnData, args_to_check: Mapping[str, Any]):
        problem = GWProblem(adata=adata_space_rotate)
        problem = problem.prepare(
            key="batch",
            x_attr={"attr": "obsm", "key": "spatial"},
            y_attr={"attr": "obsm", "key": "spatial"},
            policy="sequential",
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
            assert el == args_to_check["linear_solver_kwargs"][arg], arg

        quad_prob = problem[key]._solver._problem
        for arg, val in quad_prob_args.items():
            assert hasattr(quad_prob, val)
            assert getattr(quad_prob, val) == args_to_check[arg]

        geom = quad_prob.geom_xx
        for arg, val in geometry_args.items():
            assert hasattr(geom, val)
            assert getattr(geom, val) == args_to_check[arg]

    @pytest.mark.fast()
    @pytest.mark.parametrize(
        ("cost_str", "cost_inst", "cost_kwargs"),
        [
            ("sq_euclidean", SqEuclidean, {}),
            ("euclidean", Euclidean, {}),
            ("cosine", Cosine, {}),
            ("pnorm_p", PNormP, {"p": 3}),
            ("sq_pnorm", SqPNorm, {"x": {"p": 3}, "y": {"p": 4}}),
            ("elastic_l1", ElasticL1, {}),
            ("elastic_stvs", ElasticSTVS, {}),
        ],
    )
    def test_prepare_costs(self, adata_time: AnnData, cost_str: str, cost_inst: Any, cost_kwargs: Mapping[str, int]):
        problem = GWProblem(adata=adata_time)
        problem = problem.prepare(
            key="time", policy="sequential", x_attr="X_pca", y_attr="X_pca", cost=cost_str, cost_kwargs=cost_kwargs
        )
        assert isinstance(problem[0, 1].x.cost, cost_inst)
        assert isinstance(problem[0, 1].y.cost, cost_inst)

        problem = problem.solve(max_iterations=2)

    @pytest.mark.parametrize("tag", ["cost_matrix", "kernel"])
    def test_set_x(self, adata_time: AnnData, tag: Literal["cost_matrix", "kernel"]):
        rng = np.random.RandomState(42)
        adata_time = adata_time[adata_time.obs["time"].isin((0, 1))].copy()
        problem = GWProblem(adata=adata_time)
        problem = problem.prepare(
            x_attr="X_pca",
            y_attr="X_pca",
            key="time",
            policy="sequential",
        )

        adata_0 = adata_time[adata_time.obs["time"] == 0]

        cm = rng.uniform(1, 10, size=(adata_0.n_obs, adata_0.n_obs))
        cost_matrix = pd.DataFrame(index=adata_0.obs_names, columns=adata_0.obs_names, data=cm)
        problem[0, 1].set_x(cost_matrix, tag=tag)
        assert isinstance(problem[0, 1].x.data_src, np.ndarray)
        assert problem[0, 1].x.data_tgt is None

        problem = problem.solve(
            max_iterations=5, scale_cost=1
        )  # TODO(@MUCDK) once fixed in OTT-JAX test for scale_cost
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
            key="time",
            policy="sequential",
        )

        adata_1 = adata_time[adata_time.obs["time"] == 1]

        cm = rng.uniform(1, 10, size=(adata_1.n_obs, adata_1.n_obs))
        cost_matrix = pd.DataFrame(index=adata_1.obs_names, columns=adata_1.obs_names, data=cm)
        problem[0, 1].set_y(cost_matrix, tag=tag)
        assert isinstance(problem[0, 1].y.data_src, np.ndarray)
        assert problem[0, 1].y.data_tgt is None

        problem = problem.solve(
            max_iterations=5, scale_cost=1
        )  # TODO(@MUCDK) once fixed in OTT-JAX test for scale_cost
        assert isinstance(problem[0, 1].y.data_src, np.ndarray)
        assert problem[0, 1].y.data_tgt is None

    @pytest.mark.fast()
    def test_prepare_different_costs(self, adata_time: AnnData):
        problem = GWProblem(adata=adata_time)
        problem = problem.prepare(
            key="time",
            policy="sequential",
            x_attr="X_pca",
            y_attr="X_pca",
            cost={"x": "euclidean", "y": "sq_euclidean"},
        )
        assert isinstance(problem[0, 1].x.cost, Euclidean)
        assert isinstance(problem[0, 1].y.cost, SqEuclidean)
