import pytest

import numpy as np

import anndata as ad

from moscot.backends.ott.nets import MLP_marginal
from moscot.backends.ott.output import NeuralDualOutput
from moscot.base.output import BaseSolverOutput
from moscot.base.problems import NeuralOTProblem
from moscot.problems.generic import NeuralProblem  # type: ignore[attr-defined]
from tests._utils import ATOL, RTOL
from tests.problems.conftest import (
    neuraldual_args_1,
    neuraldual_args_2,
    neuraldual_solver_args,
)


class TestNeuralProblem:
    @pytest.mark.fast()
    def test_prepare(self, adata_time: ad.AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = NeuralProblem(adata=adata_time)

        assert len(problem) == 0
        assert problem.problems == {}
        assert problem.solutions == {}

        problem = problem.prepare(key="time", joint_attr="X_pca", policy="sequential")

        assert isinstance(problem.problems, dict)
        assert len(problem.problems) == len(expected_keys)

        for key in problem:
            assert key in expected_keys
            assert isinstance(problem[key], NeuralOTProblem)

    def test_solve_balanced_no_baseline(self, adata_time: ad.AnnData):  # type: ignore[no-untyped-def]
        expected_keys = [(0, 1), (1, 2)]
        problem = NeuralProblem(adata=adata_time)
        problem = problem.prepare(key="time", joint_attr="X_pca")
        problem = problem.solve(**neuraldual_args_1)

        for key, subsol in problem.solutions.items():
            assert isinstance(subsol, BaseSolverOutput)
            assert key in expected_keys

    def test_solve_unbalanced_with_baseline(self, adata_time: ad.AnnData):
        expected_keys = [(0, 1), (1, 2)]
        problem = NeuralProblem(adata=adata_time)
        problem = problem.prepare(key="time", joint_attr="X_pca")
        problem = problem.solve(**neuraldual_args_2)

        for key, subsol in problem.solutions.items():
            assert isinstance(subsol, BaseSolverOutput)
            assert key in expected_keys

    def test_reproducibility(self, adata_time: ad.AnnData):
        pc_tzero = adata_time[adata_time.obs["time"] == 0].obsm["X_pca"]
        problem_one = NeuralProblem(adata=adata_time)
        problem_one = problem_one.prepare(key="time", joint_attr="X_pca")
        problem_one = problem_one.solve(**neuraldual_args_1)

        problem_two = NeuralProblem(adata=adata_time)
        problem_two = problem_one.prepare("time", joint_attr="X_pca")
        problem_two = problem_one.solve(**neuraldual_args_1)

        for key in problem_one.solutions:
            assert np.allclose(
                problem_one[key].solution.push(pc_tzero),
                problem_two[key].solution.push(pc_tzero),
                rtol=RTOL,
                atol=ATOL,
            )
            assert np.allclose(
                problem_one[key].solution.pull(pc_tzero),
                problem_two[key].solution.pull(pc_tzero),
                rtol=RTOL,
                atol=ATOL,
            )

    def test_pass_arguments(self, adata_time: ad.AnnData):
        problem = NeuralProblem(adata=adata_time)
        adata_time = adata_time[adata_time.obs["time"].isin((0, 1))]
        problem = problem.prepare(key="time", joint_attr="X_pca")
        problem = problem.solve(**neuraldual_args_1)

        key = (0, 1)
        solver = problem[key].solver.solver
        assert solver.cond_dim == 0
        for arg, val in neuraldual_solver_args.items():
            assert hasattr(solver, val)
            el = getattr(solver, val)[0] if isinstance(getattr(solver, val), tuple) else getattr(solver, val)
            assert el == neuraldual_args_1[arg]

    def test_learning_rescaling_factors(self, adata_time: ad.AnnData):
        hidden_dim = 10
        problem = NeuralProblem(adata=adata_time)
        mlp_eta = MLP_marginal(hidden_dim)
        mlp_xi = MLP_marginal(hidden_dim)
        adata_time = adata_time[adata_time.obs["time"].isin((0, 1))]
        problem = problem.prepare(key="time", joint_attr="X_pca")
        problem = problem.solve(mlp_eta=mlp_eta, mlp_xi=mlp_xi, **neuraldual_args_2)
        assert isinstance(problem[0, 1].solution, BaseSolverOutput)
        assert isinstance(problem[0, 1].solution, NeuralDualOutput)

        array = adata_time.obsm["X_pca"]
        learnt_eta = problem[0, 1].solution.evaluate_a(array)
        learnt_xi = problem[0, 1].solution.evaluate_b(array)
        assert learnt_eta.shape == (array.shape[0], 1)
        assert learnt_xi.shape == (array.shape[0], 1)
        assert np.sum(np.isnan(learnt_eta)) == 0
        assert np.sum(np.isnan(learnt_xi)) == 0
