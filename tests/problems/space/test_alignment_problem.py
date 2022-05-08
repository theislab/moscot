import pytest

import numpy as np

from anndata import AnnData

from moscot.problems.space import AlignmentProblem
from moscot.problems._base_problem import OTProblem


class TestAlignmentProblem:
    def test_prepare_sequential(
        self,
        adata_space_rotate: AnnData,
    ):
        expected_keys = [("0", "1"), ("1", "2")]
        n_obs = adata_space_rotate.shape[0] // 3
        n_var = adata_space_rotate.shape[1]
        problem = AlignmentProblem(adata=adata_space_rotate)
        assert len(problem) == 0
        assert problem.problems is None
        assert problem.solutions is None

        problem = problem.prepare(batch_key="batch")
        for prob_key, exp_key in zip(problem, expected_keys):
            assert prob_key == exp_key
            assert isinstance(problem[prob_key], OTProblem)
            assert problem[prob_key].shape == (n_obs, n_obs)
            assert problem[prob_key].x.data.shape == problem[prob_key].y.data.shape == (n_obs, 2)
            assert problem[prob_key].xy[0].data.shape == problem[prob_key].xy[1].data.shape == (n_obs, n_var)

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
            assert isinstance(problem[prob_key], OTProblem)

    @pytest.mark.parametrize(
        ("epsilon", "alpha", "rank"),
        [(1, 0.9, None), (1, 0.5, None), (0.1, 0.1, None)],  # TODO(giovp): rank doesn't work?
    )  # can't set rank
    def test_solve_balance(self, adata_space_rotate: AnnData, epsilon: float, alpha: float, rank: int):

        problem = (
            AlignmentProblem(adata=adata_space_rotate)
            .prepare(batch_key="batch")
            .solve(epsilon=epsilon, alpha=alpha, rank=rank)
        )

        epsilon = 1.0 if epsilon is None else epsilon
        False if rank is None else True
        rank = -1 if rank is None else rank
        for prob_key in problem:
            assert problem[prob_key].solution.rank == rank
            assert problem[prob_key].solution.converged

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

    def test_analysis(self, adata_space_rotate: AnnData):

        adata_ref = adata_space_rotate.copy()
        problem = AlignmentProblem(adata=adata_ref).prepare(batch_key="batch").solve()
        categories = adata_space_rotate.obs.batch.cat.categories

        for ref in categories:
            problem.align(reference=ref, mode="affine")
            problem.align(reference=ref, mode="warp")
            for c1, c2 in zip(categories, categories[1:]):
                np.testing.assert_array_almost_equal(
                    adata_ref[adata_ref.obs.batch == c1].obsm["spatial_warp"],
                    adata_ref[adata_ref.obs.batch == c2].obsm["spatial_warp"],
                )
                np.testing.assert_array_almost_equal(
                    adata_ref[adata_ref.obs.batch == c1].obsm["spatial_affine"],
                    adata_ref[adata_ref.obs.batch == c2].obsm["spatial_affine"],
                )
            assert adata_ref.obsm["spatial_warp"].shape == adata_space_rotate.obsm["spatial"].shape
