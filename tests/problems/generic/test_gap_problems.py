import pytest

import numpy as np

import anndata as ad

from moscot.base.output import BaseSolverOutput
from moscot.base.problems import NeuralOTProblem
from moscot.problems.generic import NeuralProblem, MGNeuralProblem  # type: ignore[attr-defined]
from tests._utils import ATOL, RTOL
from tests.problems.conftest import (
    neuraldual_args_1,
    neuraldual_args_2,
    neuraldual_solver_args,
)


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