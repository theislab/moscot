from typing import Any, Mapping

import anndata as ad 
import numpy as np
import pandas as pd
import pytest

from moscot.problems.base import CondOTProblem
from moscot.solvers._output import BaseSolverOutput
from moscot.problems.generic import ConditionalNeuralProblem  # type: ignore[attr-defined]
from tests.problems.conftest import (
    neuraldual_args_1,
    neuraldual_args_2,
    neuraldual_solver_args,
)
from tests._utils import ATOL, RTOL

class TestConditionalNeuralProblem:
    @pytest.mark.fast()
    def test_prepare(self, adata_time: ad.AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = ConditionalNeuralProblem(adata=adata_time)

        assert len(problem) == 0
        assert problem.problems == {}
        assert problem.solutions == {}

        problem = problem.prepare(key="time", joint_attr="X_pca", policy="sequential")
        assert isinstance(problem, CondOTProblem)

    def test_solve_balanced_no_baseline(self, adata_time: ad.AnnData):  # type: ignore[no-untyped-def]
        expected_keys = [(0, 1), (1, 2)]
        problem = ConditionalNeuralProblem(adata=adata_time)
        problem = problem.prepare(time_key="time", joint_attr="X_pca")
        problem = problem.solve(**neuraldual_args_1)

        for key, subsol in problem.solutions.items():
            assert isinstance(subsol, BaseSolverOutput)
            assert key in expected_keys

    def test_solve_unbalanced_with_baseline(self, adata_time: ad.AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = ConditionalNeuralProblem(adata=adata_time)
        problem = problem.prepare(time_key="time", joint_attr="X_pca")
        problem = problem.solve(**neuraldual_args_2)

        for key, subsol in problem.solutions.items():
            assert isinstance(subsol, BaseSolverOutput)
            assert key in expected_keys

    def test_reproducibility(self, adata_time: ad.AnnData):
        pc_tzero = adata_time[adata_time.obs["time"] == 0].obsm["X_pca"]
        problem_one = ConditionalNeuralProblem(adata=adata_time)
        problem_one = problem_one.prepare(time_key="time", joint_attr="X_pca")
        problem_one = problem_one.solve(**neuraldual_args_1)

        problem_two = ConditionalNeuralProblem(adata=adata_time)
        problem_two = problem_one.prepare("time", joint_attr="X_pca")
        problem_two = problem_one.solve(**neuraldual_args_1)
        assert np.allclose(
            problem_one.solution.push(pc_tzero, 0),
            problem_two.solution.push(pc_tzero, 0),
            rtol=RTOL,
            atol=ATOL,
        )
        assert np.allclose(
            problem_one.solution.pull(pc_tzero, 0),
            problem_two.solution.pull(pc_tzero, 0),
            rtol=RTOL,
            atol=ATOL,
        )
    
    def test_pass_arguments(self, adata_time: ad.AnnData):
        problem = ConditionalNeuralProblem(adata=adata_time)
        adata_time = adata_time[adata_time.obs["time"].isin((0, 1))]
        problem = problem.prepare(key="time", joint_attr="X_pca")
        problem = problem.solve(**neuraldual_args_1)

        key = (0, 1)
        solver = problem[key].solver.solver
        assert solver.conditional == True
        for arg, val in neuraldual_solver_args.items():
            assert hasattr(solver, val)
            el = getattr(solver, val)[0] if isinstance(getattr(solver, val), tuple) else getattr(solver, val)
            assert el == neuraldual_args_1[arg]