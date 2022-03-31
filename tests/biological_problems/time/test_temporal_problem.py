import pytest

import numpy as np

from anndata import AnnData

from moscot.backends.ott import SinkhornSolver
from moscot.solvers._output import BaseSolverOutput
from moscot.problems.time._lineage import TemporalProblem, TemporalBaseProblem


class TestTemporalProblem:
    # TODO(@MUCDK) add tests for marginals
    def test_prepare(self, adata_time: AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = TemporalProblem(adata=adata_time, solver=SinkhornSolver())

        assert len(problem) == 0
        assert problem.problems is None
        assert problem.solution is None

        problem = problem.prepare(
            time_key="time",
            axis="obs",
            policy="sequential",
        )

        for key, subprob in problem:
            assert isinstance(subprob, TemporalBaseProblem)
            assert key in expected_keys

    def test_solve_balanced(self, adata_time: AnnData):
        eps = 0.5
        expected_keys = [(0, 1), (1, 2)]
        problem = TemporalProblem(adata=adata_time, solver=SinkhornSolver())
        problem = problem.prepare("time")
        problem = problem.solve(epsilon=eps)

        for key, subsol in problem.solution.items():
            assert isinstance(subsol, BaseSolverOutput)
            assert key in expected_keys

    def test_solve_unbalanced(self, adata_time: AnnData):
        taus = [9e-1, 1e-2]
        a = b = np.ones(96)
        problem1 = TemporalProblem(adata=adata_time, solver=SinkhornSolver())
        problem2 = TemporalProblem(adata=adata_time, solver=SinkhornSolver())
        problem1 = problem1.prepare("time", a=a, b=b)
        problem2 = problem2.prepare("time", a=a, b=b)
        problem1 = problem1.solve(tau_a=taus[0], tau_b=taus[0])
        problem2 = problem2.solve(tau_a=taus[1], tau_b=taus[1])

        assert problem1[0, 1].a is not None
        assert problem1[0, 1].b is not None
        assert problem2[0, 1].a is not None
        assert problem2[0, 1].b is not None

        div1 = np.linalg.norm(problem1[0, 1].a[:, -1] - np.ones(len(problem1[0, 1].a[:, -1])))
        div2 = np.linalg.norm(problem2[0, 1].a[:, -1] - np.ones(len(problem2[0, 1].a[:, -1])))
        assert div1 <= div2

    @pytest.mark.parametrize(
        "n_iters", [3]
    )  # TODO(@MUCDK) as soon as @michalk8 unified warnings/errors test for negative value
    def test_multiple_iterations(self, adata_time: AnnData, n_iters: int):
        problem = TemporalProblem(adata=adata_time, solver=SinkhornSolver())
        problem = problem.prepare("time")
        problem = problem.solve(n_iters=n_iters)

        assert problem[0, 1].growth_rates.shape[1] == n_iters + 1
        assert np.all(
            problem[0, 1].growth_rates[:, 0] == np.ones(len(problem[0, 1].a[:, -1])) / len(problem[0, 1].a[:, -1])
        )
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_almost_equal,
            problem[0, 1].growth_rates[:, 0],
            problem[0, 1].growth_rates[:, 1],
        )
