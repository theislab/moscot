from typing import Any, Literal, Mapping

import pytest
from anndata import AnnData

import numpy as np
import pandas as pd

from moscot.base.output import BaseSolverOutput
from moscot.problems.base import OTProblem
from moscot.problems.generic import SinkhornProblem
from tests.problems.conftest import (
    geometry_args,
    lin_prob_args,
    lr_pointcloud_args,
    lr_sinkhorn_solver_args,
    pointcloud_args,
    sinkhorn_args_1,
    sinkhorn_args_2,
    sinkhorn_solver_args,
)


class TestSinkhornProblem:
    @pytest.mark.fast()
    def test_prepare(self, adata_time: AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = SinkhornProblem(adata=adata_time)

        assert len(problem) == 0
        assert problem.problems == {}
        assert problem.solutions == {}

        problem = problem.prepare(
            key="time",
            policy="sequential",
        )

        assert isinstance(problem.problems, dict)
        assert len(problem.problems) == len(expected_keys)

        for key in problem:
            assert key in expected_keys
            assert isinstance(problem[key], OTProblem)

    def test_solve_balanced(self, adata_time: AnnData):
        eps = 0.5
        expected_keys = [(0, 1), (1, 2)]
        problem = SinkhornProblem(adata=adata_time)
        problem = problem.prepare(key="time")
        problem = problem.solve(epsilon=eps)

        for key, subsol in problem.solutions.items():
            assert isinstance(subsol, BaseSolverOutput)
            assert key in expected_keys

    @pytest.mark.parametrize("method", ["fischer", "perm_test"])
    def test_compute_feature_correlation(self, adata_time: AnnData, method: str):
        problem = SinkhornProblem(adata=adata_time)
        problem = problem.prepare(key="time")
        problem = problem.solve()
        assert problem[0, 1].solution.converged

        key_added = "test_push"
        problem.push(source=0, target=1, data="celltype", subset="A", key_added=key_added)
        feature_correlation = problem.compute_feature_correlation(key_added, significance_method=method)

        assert isinstance(feature_correlation, pd.DataFrame)
        suffix = ["_corr", "_pval", "_qval", "_ci_low", "_ci_high"]
        assert list(feature_correlation.columns) == [key_added + suf for suf in suffix]
        assert feature_correlation.isna().sum().sum() == 0

    @pytest.mark.parametrize("tag", ["cost_matrix", "kernel"])
    def test_set_xy(self, adata_time: AnnData, tag: Literal["cost_matrix", "kernel"]):
        rng = np.random.RandomState(42)
        adata_time = adata_time[adata_time.obs["time"].isin((0, 1))].copy()
        problem = SinkhornProblem(adata=adata_time)
        problem = problem.prepare(
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

        problem = problem.solve(
            max_iterations=5, scale_cost=1
        )  # TODO(@MUCDK) once fixed in OTT-JAX test for scale_cost
        assert isinstance(problem[0, 1].xy.data_src, np.ndarray)
        assert problem[0, 1].xy.data_tgt is None

    @pytest.mark.parametrize("args_to_check", [sinkhorn_args_1, sinkhorn_args_2])
    def test_pass_arguments(self, adata_time: AnnData, args_to_check: Mapping[str, Any]):
        adata_time = adata_time[adata_time.obs["time"].isin((0, 1))].copy()
        problem = SinkhornProblem(adata=adata_time)
        problem = problem.prepare(
            key="time",
            policy="sequential",
        )

        problem = problem.solve(**args_to_check)

        solver = problem[(0, 1)].solver.solver
        args = sinkhorn_solver_args if args_to_check["rank"] == -1 else lr_sinkhorn_solver_args
        for arg, val in args.items():
            assert hasattr(solver, val)
            el = getattr(solver, val)[0] if isinstance(getattr(solver, val), tuple) else getattr(solver, val)
            assert el == args_to_check[arg]

        lin_prob = problem[(0, 1)]._solver._problem
        for arg, val in lin_prob_args.items():
            assert hasattr(lin_prob, val)
            el = getattr(lin_prob, val)[0] if isinstance(getattr(lin_prob, val), tuple) else getattr(lin_prob, val)
            assert el == args_to_check[arg]

        geom = lin_prob.geom
        for arg, val in geometry_args.items():
            assert hasattr(geom, val)
            el = getattr(geom, val)[0] if isinstance(getattr(geom, val), tuple) else getattr(geom, val)
            assert el == args_to_check[arg]

        args = pointcloud_args if args_to_check["rank"] == -1 else lr_pointcloud_args
        for arg, val in args.items():
            el = getattr(geom, val)[0] if isinstance(getattr(geom, val), tuple) else getattr(geom, val)
            assert hasattr(geom, val)
            if arg == "cost":
                assert type(el) == type(args_to_check[arg])  # noqa: E721
            else:
                assert el == args_to_check[arg]
