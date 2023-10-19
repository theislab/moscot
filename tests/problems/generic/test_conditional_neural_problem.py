import pytest
import optax
import jax.numpy as jnp
import numpy as np

import anndata as ad

from moscot.backends.ott._icnn import ICNN
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
        problem = problem.solve(cond_dim=1, **neuraldual_args_1)
        assert isinstance(problem.solution, BaseSolverOutput)

    def test_solve_unbalanced_with_baseline(self, adata_time: ad.AnnData):
        problem = ConditionalNeuralProblem(adata=adata_time)
        problem = problem.prepare(key="time", joint_attr="X_pca")
        problem = problem.solve(cond_dim=1, **neuraldual_args_2)
        assert isinstance(problem.solution, BaseSolverOutput)

    def test_reproducibility(self, adata_time: ad.AnnData):
        pc_tzero = adata_time[adata_time.obs["time"] == 0].obsm["X_pca"]
        problem_one = ConditionalNeuralProblem(adata=adata_time)
        problem_one = problem_one.prepare(key="time", joint_attr="X_pca")
        problem_one = problem_one.solve(**neuraldual_args_1, cond_dim=1)

        problem_two = ConditionalNeuralProblem(adata=adata_time)
        problem_two = problem_one.prepare("time", joint_attr="X_pca")
        problem_two = problem_one.solve(**neuraldual_args_1, cond_dim=1)
        assert np.allclose(
            problem_one.solution.push(jnp.array([0]), pc_tzero),
            problem_two.solution.push(jnp.array([0]), pc_tzero),
            rtol=RTOL,
            atol=ATOL,
        )
        assert np.allclose(
            problem_one.solution.pull(jnp.array([0]), pc_tzero),
            problem_two.solution.pull(jnp.array([0]), pc_tzero),
            rtol=RTOL,
            atol=ATOL,
        )

    def test_pass_arguments(self, adata_time: ad.AnnData):
        problem = ConditionalNeuralProblem(adata=adata_time)
        adata_time = adata_time[adata_time.obs["time"].isin((0, 1))]
        problem = problem.prepare(key="time", joint_attr="X_pca")
        problem = problem.solve(cond_dim=1, **neuraldual_args_1)

        solver = problem.solver.solver
        assert solver.cond_dim > 0
        for arg, val in neuraldual_solver_args.items():
            assert hasattr(solver, val)
            el = getattr(solver, val)[0] if isinstance(getattr(solver, val), tuple) else getattr(solver, val)
            assert el == neuraldual_args_1[arg]

    def test_pass_custom_mlps(self, adata_time: ad.AnnData):
        problem = ConditionalNeuralProblem(adata=adata_time)
        adata_time = adata_time[adata_time.obs["time"].isin((0, 1))]
        problem = problem.prepare(key="time", joint_attr="X_pca")
        input_dim = adata_time.obsm["X_pca"].shape[1]
        custom_f = ICNN([3, 3], input_dim=input_dim, cond_dim=1)
        custom_g = ICNN([3, 3], input_dim=input_dim, cond_dim=1)

        problem = problem.solve(iterations=2, f=custom_f, g=custom_g, cond_dim=1)
        assert problem.solver.solver.f == custom_f
        assert problem.solver.solver.g == custom_g

    def test_pass_custom_optimizers(self, adata_time: ad.AnnData):
        problem = ConditionalNeuralProblem(adata=adata_time)
        adata_time = adata_time[adata_time.obs["time"].isin((0, 1))]
        problem = problem.prepare(key="time", joint_attr="X_pca")
        custom_opt_f = optax.adagrad(1e-4)
        custom_opt_g = optax.adagrad(1e-3)

        problem = problem.solve(iterations=2, opt_f=custom_opt_f, opt_g=custom_opt_g, cond_dim=1)
        