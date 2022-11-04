from typing import Any, Mapping

import pytest

from anndata import AnnData

from moscot.problems.base import OTProblem  # type: ignore[attr-defined]
from moscot.solvers._output import BaseSolverOutput
from moscot.problems.generic import SinkhornProblem  # type: ignore[attr-defined]
from tests.problems.conftest import (
    geometry_args,
    lin_prob_args,
    pointcloud_args,
    sinkhorn_args_1,
    sinkhorn_args_2,
    sinkhorn_solver_args,
    lr_sinkhorn_solver_args,
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

    def test_solve_balanced(self, adata_time: AnnData):  # type: ignore[no-untyped-def]
        eps = 0.5
        expected_keys = [(0, 1), (1, 2)]
        problem = SinkhornProblem(adata=adata_time)
        problem = problem.prepare(key="time")
        problem = problem.solve(epsilon=eps)

        for key, subsol in problem.solutions.items():
            assert isinstance(subsol, BaseSolverOutput)
            assert key in expected_keys

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

        for arg, val in pointcloud_args.items():
            el = getattr(geom, val)[0] if isinstance(getattr(geom, val), tuple) else getattr(geom, val)
            assert hasattr(geom, val)
            if arg == "cost":
                assert type(el) == type(args_to_check[arg])  # noqa: E721
            else:
                assert el == args_to_check[arg]
