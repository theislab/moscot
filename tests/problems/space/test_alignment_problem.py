from typing import Any, Mapping, Optional
from pathlib import Path

import pytest

import numpy as np

from anndata import AnnData

from moscot.problems.space import AlignmentProblem

SOLUTIONS_PATH = Path("./tests/data/alignment_solutions.pkl")  # base is moscot


class TestAlignmentProblem:
    @pytest.mark.fast()
    @pytest.mark.parametrize(
        "joint_attr", [{"x_attr": "X", "y_attr": "X"}]
    )  # TODO(giovp): check that callback is correct
    def test_prepare_sequential(self, adata_space_rotate: AnnData, joint_attr: Optional[Mapping[str, Any]]):
        n_obs = adata_space_rotate.shape[0] // 3  # adata is made of 3 datasets
        n_var = adata_space_rotate.shape[1]
        expected_keys = {("0", "1"), ("1", "2")}
        ap = AlignmentProblem(adata=adata_space_rotate)
        assert len(ap) == 0
        assert ap.problems == {}
        assert ap.solutions == {}

        ap = ap.prepare(batch_key="batch", joint_attr=joint_attr)
        assert len(ap) == 2

        for prob_key in expected_keys:
            assert isinstance(ap[prob_key], ap._base_problem_type)
            assert ap[prob_key].shape == (n_obs, n_obs)
            assert ap[prob_key].x.data.shape == ap[prob_key].y.data.shape == (n_obs, 2)
            assert ap[prob_key].xy.data.shape == ap[prob_key].xy.data_y.shape == (n_obs, n_var)

    @pytest.mark.fast()
    @pytest.mark.parametrize("reference", ["0", "1", "2"])
    def test_prepare_star(self, adata_space_rotate: AnnData, reference: str):
        ap = AlignmentProblem(adata=adata_space_rotate)
        assert len(ap) == 0
        assert ap.problems == {}
        assert ap.solutions == {}
        ap = ap.prepare(batch_key="batch", policy="star", reference=reference)
        for prob_key in ap:
            _, ref = prob_key
            assert ref == reference
            assert isinstance(ap[prob_key], ap._base_problem_type)

    @pytest.mark.parametrize(
        ("epsilon", "alpha", "rank"),
        [(1, 0.9, -1), (1, 0.5, 10), (0.1, 0.1, -1)],
    )
    def test_solve_balanced(self, adata_space_rotate: AnnData, epsilon: float, alpha: float, rank: int):
        ap = (
            AlignmentProblem(adata=adata_space_rotate)
            .prepare(batch_key="batch")
            .solve(epsilon=epsilon, alpha=alpha, rank=rank)
        )
        for prob_key in ap:
            assert ap[prob_key].solution.rank == rank
            assert ap[prob_key].solution.converged

        assert np.allclose(*(sol.cost for sol in ap.solutions.values()))
        assert np.all([sol.converged for sol in ap.solutions.values()])
        assert np.all([np.all(~np.isnan(sol.transport_matrix)) for sol in ap.solutions.values()])

    def test_solve_unbalanced(self, adata_space_rotate: AnnData):  # unclear usage yet
        tau_a, tau_b = [0.8, 1]
        marg_a = "a"
        marg_b = "b"
        adata_space_rotate.obs[marg_a] = adata_space_rotate.obs[marg_b] = np.ones(300)
        ap = (
            AlignmentProblem(adata=adata_space_rotate)
            .prepare(batch_key="batch", a=marg_a, b=marg_b)
            .solve(tau_a=tau_a, tau_b=tau_b, scale_cost=False)
        )
        assert np.all([sol.a is not None for sol in ap.solutions.values()])
        assert np.all([sol.b is not None for sol in ap.solutions.values()])
        assert np.all([sol.converged for sol in ap.solutions.values()])
        assert np.allclose(*(sol.cost for sol in ap.solutions.values()))
