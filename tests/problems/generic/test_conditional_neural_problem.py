import pytest

import numpy as np

import anndata as ad

from moscot.base.output import BaseSolverOutput
from moscot.base.problems import CondOTProblem
from moscot.problems.generic import (
    ConditionalNeuralProblem,  # type: ignore[attr-defined]
)
from tests._utils import ATOL, RTOL
from tests.problems.conftest import (
    neuraldual_args_1,
    neuraldual_args_2,
    neuraldual_solver_args,
)


class TestConditionalNeuralProblem:
    @pytest.mark.fast()
    def test_prepare(self, adata_time: ad.AnnData):
        problem = ConditionalNeuralProblem(adata=adata_time)
        problem = problem.prepare(key="time", joint_attr="X_pca")
        assert isinstance(problem, CondOTProblem)

    def test_solve_balanced_no_baseline(self, adata_time: ad.AnnData):  # type: ignore[no-untyped-def]
        problem = ConditionalNeuralProblem(adata=adata_time)
        problem = problem.prepare(key="time", joint_attr="X_pca")
        problem = problem.solve(**neuraldual_args_1)
        assert isinstance(problem.solution, BaseSolverOutput)

    def test_solve_unbalanced_with_baseline(self, adata_time: ad.AnnData):
        problem = ConditionalNeuralProblem(adata=adata_time)
        problem = problem.prepare(key="time", joint_attr="X_pca")
        problem = problem.solve(**neuraldual_args_2)
        assert isinstance(problem.solution, BaseSolverOutput)

    def test_reproducibility(self, adata_time: ad.AnnData):
        pc_tzero = adata_time[adata_time.obs["time"] == 0].obsm["X_pca"]
        problem_one = ConditionalNeuralProblem(adata=adata_time)
        problem_one = problem_one.prepare(key="time", joint_attr="X_pca")
        problem_one = problem_one.solve(**neuraldual_args_1)

        problem_two = ConditionalNeuralProblem(adata=adata_time)
        problem_two = problem_one.prepare("time", joint_attr="X_pca")
        problem_two = problem_one.solve(**neuraldual_args_1)
        assert np.allclose(
            problem_one.solution.push(0, pc_tzero),
            problem_two.solution.push(0, pc_tzero),
            rtol=RTOL,
            atol=ATOL,
        )
        assert np.allclose(
            problem_one.solution.pull(0, pc_tzero),
            problem_two.solution.pull(0, pc_tzero),
            rtol=RTOL,
            atol=ATOL,
        )

    def test_pass_arguments(self, adata_time: ad.AnnData):
        problem = ConditionalNeuralProblem(adata=adata_time)
        adata_time = adata_time[adata_time.obs["time"].isin((0, 1))]
        problem = problem.prepare(key="time", joint_attr="X_pca")
        problem = problem.solve(**neuraldual_args_1)
      
        solver = problem.solver.solver
        assert solver.conditional is True
        for arg, val in neuraldual_solver_args.items():
            assert hasattr(solver, val)
            el = getattr(solver, val)[0] if isinstance(getattr(solver, val), tuple) else getattr(solver, val)
            assert el == neuraldual_args_1[arg]
