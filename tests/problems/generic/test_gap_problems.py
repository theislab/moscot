import pytest

import anndata as ad

from moscot.base.problems import NeuralOTProblem
from moscot.problems.generic import MGNeuralProblem  # type: ignore[attr-defined]


class TestMGNeuralProblem:
    @pytest.mark.fast()
    def test_prepare(self, adata_time: ad.AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = MGNeuralProblem(adata=adata_time)

        assert len(problem) == 0
        assert problem.problems == {}
        assert problem.solutions == {}

        problem = problem.prepare(key="time", joint_attr="X_pca", policy="sequential")

        assert isinstance(problem.problems, dict)
        assert len(problem.problems) == len(expected_keys)

        for key in problem:
            assert key in expected_keys
            assert isinstance(problem[key], NeuralOTProblem)

    def test_solve(self, adata_time: ad.AnnData):
        pass
