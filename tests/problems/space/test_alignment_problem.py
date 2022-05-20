from pathlib import Path

import pytest

import numpy as np

from anndata import AnnData

from moscot.problems.space import AlignmentProblem
from moscot.problems._base_problem import OTProblem

SOLUTIONS_PATH = Path("./tests/data/alignment_solutions.pkl")  # base is moscot


class TestAlignmentProblem:
    def test_prepare_sequential(
        self,
        adata_space_rotate: AnnData,
    ):
        expected_keys = [("0", "1"), ("1", "2")]
        n_obs = adata_space_rotate.shape[0] // 3
        n_var = adata_space_rotate.shape[1]
        ap = AlignmentProblem(adata=adata_space_rotate)
        assert len(ap) == 0
        assert ap.problems is None
        assert ap.solutions is None

        ap = ap.prepare(batch_key="batch")
        for prob_key, exp_key in zip(ap, expected_keys):
            assert prob_key == exp_key
            assert isinstance(ap[prob_key], OTProblem)
            assert ap[prob_key].shape == (n_obs, n_obs)
            assert ap[prob_key].x.data.shape == ap[prob_key].y.data.shape == (n_obs, 2)
            assert ap[prob_key].xy[0].data.shape == ap[prob_key].xy[1].data.shape == (n_obs, n_var)

    @pytest.mark.parametrize("reference", ["0", "1", "2"])
    def test_prepare_star(self, adata_space_rotate: AnnData, reference: str):
        ap = AlignmentProblem(adata=adata_space_rotate)
        assert len(ap) == 0
        assert ap.problems is None
        assert ap.solutions is None
        ap = ap.prepare(batch_key="batch", policy="star", reference=reference)
        for prob_key in ap:
            _, ref = prob_key
            assert ref == reference
            assert isinstance(ap[prob_key], OTProblem)

    @pytest.mark.parametrize(
        ("epsilon", "alpha", "rank"),
        [(1, 0.9, None), (1, 0.5, None), (0.1, 0.1, None)],  # TODO(giovp): rank doesn't work?
    )  # can't set rank
    def test_solve_balance(self, adata_space_rotate: AnnData, epsilon: float, alpha: float, rank: int):
        ap = (
            AlignmentProblem(adata=adata_space_rotate)
            .prepare(batch_key="batch")
            .solve(epsilon=epsilon, alpha=alpha, rank=rank)
        )
        epsilon = 1.0 if epsilon is None else epsilon
        False if rank is None else True
        rank = -1 if rank is None else rank
        for prob_key in ap:
            assert ap[prob_key].solution.rank == rank
            assert ap[prob_key].solution.converged

        assert np.allclose(*(sol.cost for sol in ap.solutions.values()))
        assert np.all([sol.converged for sol in ap.solutions.values()])
        assert np.all([np.all(~np.isnan(sol.transport_matrix)) for sol in ap.solutions.values()])

    def test_solve_unbalanced(self, adata_space_rotate: AnnData):  # unclear usage yet
        tau_a, tau_b = [1, 1.2]
        a = b = np.ones(100)
        ap = (
            AlignmentProblem(adata=adata_space_rotate)
            .prepare(batch_key="batch", a=a, b=b)
            .solve(tau_a=tau_a, tau_b=tau_b, scale_cost=False)
        )
        assert np.all([sol.a is not None for sol in ap.solutions.values()])
        assert np.all([sol.b is not None for sol in ap.solutions.values()])
        assert np.all([sol.converged for sol in ap.solutions.values()])
        # assert np.allclose(*[sol.cost for sol in problem.solutions.values()]) # nan returned
