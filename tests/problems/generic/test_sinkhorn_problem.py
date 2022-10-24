from typing import Any, Mapping

import pytest

from anndata import AnnData

from moscot.problems.base import OTProblem
from moscot.solvers._output import BaseSolverOutput
from moscot.problems.generic import SinkhornProblem
from tests.problems.conftest import (  # type: ignore[attr-defined]
    geometry_args,
    lin_prob_args,
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

    @pytest.mark.parametrize("args_to_check", [sinkhorn_args_1, sinkhorn_args_2])
    def test_pass_arguments(self, adata_time: AnnData, args_to_check: Mapping[str, Any]):
        problem = SinkhornProblem(adata=adata_time)

        problem = problem.prepare(
            key="time",
            policy="sequential",
            filter=[(0, 1)],
        )

        problem = problem.solve(**args_to_check)

        solver = problem[(0, 1)]._solver._solver
        for arg in sinkhorn_solver_args:
            assert hasattr(solver, sinkhorn_solver_args[arg])
            el = (
                getattr(solver, sinkhorn_solver_args[arg])[0]
                if isinstance(getattr(solver, sinkhorn_solver_args[arg]), tuple)
                else getattr(solver, sinkhorn_solver_args[arg])
            )
            assert el == args_to_check[arg]

        lin_prob = problem[(0, 1)]._solver._problem
        for arg in lin_prob_args:
            assert hasattr(lin_prob, lin_prob_args[arg])
            el = (
                getattr(lin_prob, lin_prob_args[arg])[0]
                if isinstance(getattr(lin_prob, lin_prob_args[arg]), tuple)
                else getattr(lin_prob, lin_prob_args[arg])
            )
            assert el == args_to_check[arg]

        geom = lin_prob.geom
        for arg in geometry_args:
            assert hasattr(geom, geometry_args[arg])
            el = (
                getattr(geom, geometry_args[arg])[0]
                if isinstance(getattr(geom, geometry_args[arg]), tuple)
                else getattr(geom, geometry_args[arg])
            )
            assert el == args_to_check[arg]

        for arg in pointcloud_args:
            el = (
                getattr(geom, pointcloud_args[arg])[0]
                if isinstance(getattr(geom, pointcloud_args[arg]), tuple)
                else getattr(geom, pointcloud_args[arg])
            )
            assert hasattr(geom, pointcloud_args[arg])
            if arg == "cost":
                assert type(el) == type(args_to_check[arg])  # noqa: E721
            else:
                assert el == args_to_check[arg]
