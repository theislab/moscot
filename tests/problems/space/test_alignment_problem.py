from typing import Union, Optional

import pytest

import numpy as np

from anndata import AnnData

from moscot.problems.space import AlignmentProblem
from moscot.problems._base_problem import GeneralProblem


class TestAlignmentProblem:
    @pytest.mark.parametrize(("epsilon", "rank"), [(None, None), (3, None), (1e-5, 10)])
    def test_prepare_sequential(
        self, adata_space_rotate: AnnData, epsilon: Optional[Union[int, float]], rank: Optional[int]
    ):
        expected_keys = [("0", "1"), ("1", "2")]
        solver_kwargs = {"epsilon": epsilon}
        if rank is not None:
            solver_kwargs["rank"] = rank
        problem = AlignmentProblem(adata=adata_space_rotate, solver_kwargs=solver_kwargs)
        assert len(problem) == 0
        assert problem.problems is None
        assert problem.solutions is None

        problem = problem.prepare(batch_key="batch")
        for prob_key, exp_key in zip(problem, expected_keys):
            assert prob_key == exp_key
            assert isinstance(problem[prob_key], GeneralProblem)
        epsilon = 1.0 if epsilon is None else epsilon
        is_low_rank = False if rank is None else True
        rank = -1 if rank is None else rank
        assert problem[prob_key].solver.epsilon == epsilon
        assert problem[prob_key].solver.rank == rank
        assert problem[prob_key].solver.is_low_rank == is_low_rank

    @pytest.mark.parametrize("reference", ["0", "1", "2"])
    def test_prepare_star(self, adata_space_rotate: AnnData, reference: str):
        problem = AlignmentProblem(adata=adata_space_rotate)
        assert len(problem) == 0
        assert problem.problems is None
        assert problem.solutions is None

        problem = problem.prepare(batch_key="batch", policy="star", reference=reference)
        for prob_key in problem:
            _, ref = prob_key
            assert ref == reference
            assert isinstance(problem[prob_key], GeneralProblem)

    @pytest.mark.parametrize(("epsilon", "alpha"), [(1, 0.9), (1, 0.5), (1e-3, 0.1)])
    def test_solve_balance(self, adata_space_rotate: AnnData, epsilon: float, alpha: float):
        problem = (
            AlignmentProblem(adata=adata_space_rotate).prepare(batch_key="batch").solve(epsilon=epsilon, alpha=alpha)
        )

        assert np.allclose(*(sol.cost for sol in problem.solutions.values()))
        assert np.all([sol.converged for sol in problem.solutions.values()])
        assert np.all([np.all(~np.isnan(sol.transport_matrix)) for sol in problem.solutions.values()])

    def test_solve_unbalance(self, adata_space_rotate: AnnData):  # unclear usage yet
        tau_a, tau_b = [1, 1.2]
        a = b = np.ones(100)
        problem = (
            AlignmentProblem(adata=adata_space_rotate)
            .prepare(batch_key="batch", a=a, b=b)
            .solve(tau_a=tau_a, tau_b=tau_b)
        )

        assert np.all([sol.a is not None for sol in problem.solutions.values()])
        assert np.all([sol.b is not None for sol in problem.solutions.values()])
        assert np.all([sol.converged for sol in problem.solutions.values()])
        # assert np.allclose(*[sol.cost for sol in problem.solutions.values()]) # nan returned
