from anndata import AnnData

from moscot.problems.space import AlignmentProblem
from moscot.problems._base_problem import GeneralProblem

EXPECTED_KEYS = [("0", "1"), ("1", "2")]


class TestAlignmentProblem:
    def test_prepare_sequential(self, adata_space_rotate: AnnData):

        problem = AlignmentProblem(adata=adata_space_rotate)

        assert len(problem) == 0
        assert problem.problems is None
        assert problem.solutions is None

        problem = problem.prepare(batch_key="batch")

        for prob_key, exp_key in zip(problem, EXPECTED_KEYS):
            assert prob_key == exp_key
            assert isinstance(problem[prob_key], GeneralProblem)
