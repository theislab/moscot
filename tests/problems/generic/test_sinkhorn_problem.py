from typing import Any, Callable, Literal, Mapping

import pytest

import numpy as np
import pandas as pd
from ott.geometry.costs import Cosine, Euclidean, PNormP, SqEuclidean, SqPNorm
from ott.solvers.linear import acceleration

from anndata import AnnData

from moscot.base.output import BaseDiscreteSolverOutput
from moscot.base.problems import OTProblem
from moscot.problems.generic import SinkhornProblem
from tests._utils import _assert_marginals_set
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
    @pytest.mark.fast
    @pytest.mark.parametrize("policy", ["sequential", "star"])
    def test_prepare(self, adata_time: AnnData, policy, marginal_keys):
        expected_keys = {"sequential": [(0, 1), (1, 2)], "star": [(1, 0), (2, 0)]}
        problem = SinkhornProblem(adata=adata_time)
        assert len(problem) == 0
        assert problem.problems == {}
        assert problem.solutions == {}

        problem = problem.prepare(key="time", policy=policy, reference=0, a=marginal_keys[0], b=marginal_keys[1])

        assert isinstance(problem.problems, dict)
        assert len(problem.problems) == len(expected_keys[policy])

        for key in problem:
            assert key in expected_keys[policy]
            assert isinstance(problem[key], OTProblem)

            _assert_marginals_set(adata_time, problem, key, marginal_keys)

    def test_solve_balanced(self, adata_time: AnnData, marginal_keys):
        eps = 0.5
        expected_keys = [(0, 1), (1, 2)]
        problem = SinkhornProblem(adata=adata_time)
        problem = problem.prepare(key="time", a=marginal_keys[0], b=marginal_keys[1])
        problem = problem.solve(epsilon=eps)

        for key, subsol in problem.solutions.items():
            assert isinstance(subsol, BaseDiscreteSolverOutput)
            assert key in expected_keys
            assert subsol.converged
            assert np.allclose(subsol.a, problem[key].a, atol=1e-5)
            assert np.allclose(subsol.b, problem[key].b, atol=1e-5)

    @pytest.mark.fast
    @pytest.mark.parametrize(
        ("cost_str", "cost_inst", "cost_kwargs"),
        [
            ("sq_euclidean", SqEuclidean, {}),
            ("euclidean", Euclidean, {}),
            ("cosine", Cosine, {}),
            ("pnorm_p", PNormP, {"p": 3}),
            ("sq_pnorm", SqPNorm, {"p": 3}),
        ],
    )
    def test_prepare_costs(self, adata_time: AnnData, cost_str: str, cost_inst: Any, cost_kwargs: Mapping[str, int]):
        problem = SinkhornProblem(adata=adata_time)
        problem = problem.prepare(
            key="time", policy="sequential", joint_attr="X_pca", cost=cost_str, cost_kwargs=cost_kwargs
        )
        if cost_kwargs:
            for k, v in cost_kwargs.items():
                assert getattr(problem[0, 1].xy.cost, k) == v

        problem = problem.solve(max_iterations=2)

    @pytest.mark.fast
    @pytest.mark.parametrize(
        ("cost_str", "cost_inst", "cost_kwargs"),
        [
            ("sq_euclidean", SqEuclidean, {}),
            ("euclidean", Euclidean, {}),
            ("cosine", Cosine, {}),
            ("pnorm_p", PNormP, {"p": 3}),
            ("sq_pnorm", SqPNorm, {"p": 3}),
        ],
    )
    def test_prepare_costs_with_callback(
        self, adata_time: AnnData, cost_str: str, cost_inst: Any, cost_kwargs: Mapping[str, int]
    ):
        problem = SinkhornProblem(adata=adata_time)
        problem = problem.prepare(
            key="time", policy="sequential", xy_callback="local-pca", cost=cost_str, cost_kwargs=cost_kwargs
        )
        if cost_kwargs:
            for k, v in cost_kwargs.items():
                assert getattr(problem[0, 1].xy.cost, k) == v

        problem = problem.solve(max_iterations=2)

    @pytest.mark.parametrize("method", ["fisher", "perm_test"])
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

        # TODO(@MUCDK) once fixed in OTT-JAX test for scale_cost
        problem = problem.solve(max_iterations=5, scale_cost=1)
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
            if arg != "initializer_kwargs":
                assert hasattr(solver, val), val
                el = getattr(solver, val)[0] if isinstance(getattr(solver, val), tuple) else getattr(solver, val)
                if arg == "initializer":
                    assert isinstance(el, Callable)
                else:
                    assert el == args_to_check[arg], arg

        lin_prob = problem[(0, 1)]._solver._problem
        for arg, val in lin_prob_args.items():
            assert hasattr(lin_prob, val), val
            el = getattr(lin_prob, val)[0] if isinstance(getattr(lin_prob, val), tuple) else getattr(lin_prob, val)
            assert el == args_to_check[arg], arg

        geom = lin_prob.geom
        for arg, val in geometry_args.items():
            assert hasattr(geom, val)
            el = getattr(geom, val)[0] if isinstance(getattr(geom, val), tuple) else getattr(geom, val)
            if arg == "epsilon":
                eps_processed = getattr(geom, val)
                assert eps_processed == args_to_check[arg], arg
            else:
                assert getattr(geom, val) == args_to_check[arg], arg
                assert el == args_to_check[arg]

        args = pointcloud_args if args_to_check["rank"] == -1 else lr_pointcloud_args
        for arg, val in args.items():
            el = getattr(geom, val)[0] if isinstance(getattr(geom, val), tuple) else getattr(geom, val)
            assert hasattr(geom, val), val
            if arg == "cost":
                assert type(el) == type(args_to_check[arg]), arg  # noqa: E721
            else:
                assert el == args_to_check[arg], arg

    @pytest.mark.parametrize(("memory", "refresh"), [(1, 1), (5, 3), (7, 5)])
    @pytest.mark.parametrize("recenter", [True, False])
    def test_passing_ott_kwargs(self, adata_time: AnnData, memory: int, refresh: int, recenter: bool):
        problem = SinkhornProblem(adata=adata_time)
        problem = problem.prepare(
            key="time",
            policy="sequential",
        )

        problem = problem.solve(
            inner_iterations=1,
            max_iterations=1,
            anderson=acceleration.AndersonAcceleration(memory=memory, refresh_every=refresh),
            recenter_potentials=recenter,
        )

        anderson = problem[0, 1].solver.solver.anderson
        assert isinstance(anderson, acceleration.AndersonAcceleration)
        assert anderson.memory == memory
        assert anderson.refresh_every == refresh

        recenter_potentials = problem[0, 1].solver.solver.recenter_potentials
        assert recenter_potentials == recenter

    @pytest.mark.parametrize("reference", [0, 1, 2])
    def test_sinkhorn_problem_star(self, adata_time: AnnData, reference: int):
        problem = SinkhornProblem(adata=adata_time)
        problem = problem.prepare(key="time", joint_attr="X_pca", policy="star", reference=reference)
        if reference == 0:
            expected_policy_graph = {(1, 0), (2, 0)}
        elif reference == 1:
            expected_policy_graph = {(0, 1), (2, 1)}
        else:
            expected_policy_graph = {(0, 2), (1, 2)}
        assert set(problem._policy._graph) == expected_policy_graph
