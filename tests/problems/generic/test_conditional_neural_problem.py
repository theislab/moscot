import optax
import pytest

import jax.numpy as jnp
import numpy as np
from ott.geometry import costs

import anndata as ad

from moscot.backends.ott.output import NeuralDualOutput
from moscot.base.output import BaseSolverOutput
from moscot.base.problems import CondOTProblem
from moscot.problems.generic import (
    ConditionalNeuralProblem,  # type: ignore[attr-defined]
)
from moscot.utils.tagged_array import DistributionCollection, DistributionContainer
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
        problem = problem.prepare(key="time", joint_attr="X_pca", conditional_attr={"attr": "obs", "key": "time"})
        assert isinstance(problem, CondOTProblem)
        assert isinstance(problem.distributions, DistributionCollection)
        assert list(problem.distributions.keys()) == [0, 1, 2]

        container = problem.distributions[0]
        n_obs_0 = adata_time[adata_time.obs["time"] == 0].n_obs
        assert isinstance(container, DistributionContainer)
        assert isinstance(container.xy, np.ndarray)
        assert container.xy.shape == (n_obs_0, 50)
        assert container.xx is None
        assert isinstance(container.conditions, np.ndarray)
        assert container.conditions.shape == (n_obs_0, 1)
        assert isinstance(container.a, np.ndarray)
        assert container.a.shape == (n_obs_0,)
        assert isinstance(container.b, np.ndarray)
        assert container.b.shape == (n_obs_0,)
        assert isinstance(container.cost_xy, costs.SqEuclidean)
        assert container.cost_xx is None

    @pytest.mark.parametrize("train_size", [0.9, 1.0])
    def test_solve_balanced_no_baseline(self, adata_time: ad.AnnData, train_size: float):  # type: ignore[no-untyped-def]  # noqa: E501
        problem = ConditionalNeuralProblem(adata=adata_time)
        problem = problem.prepare(key="time", joint_attr="X_pca", conditional_attr={"attr": "obs", "key": "time"})
        problem = problem.solve(train_size=train_size, **neuraldual_args_1)
        assert isinstance(problem.solution, BaseSolverOutput)

    def test_solve_unbalanced_with_baseline(self, adata_time: ad.AnnData):
        problem = ConditionalNeuralProblem(adata=adata_time)
        problem = problem.prepare(key="time", joint_attr="X_pca", conditional_attr={"attr": "obs", "key": "time"})
        problem = problem.solve(**neuraldual_args_2)
        assert isinstance(problem.solution, BaseSolverOutput)

    def test_reproducibility(self, adata_time: ad.AnnData):
        pc_tzero = adata_time[adata_time.obs["time"] == 0].obsm["X_pca"]
        problem_one = ConditionalNeuralProblem(adata=adata_time)
        problem_one = problem_one.prepare(
            key="time", joint_attr="X_pca", conditional_attr={"attr": "obs", "key": "time"}
        )
        problem_one = problem_one.solve(**neuraldual_args_1)

        problem_two = ConditionalNeuralProblem(adata=adata_time)
        problem_two = problem_one.prepare("time", joint_attr="X_pca", conditional_attr={"attr": "obs", "key": "time"})
        problem_two = problem_one.solve(**neuraldual_args_1)
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
        problem = problem.prepare(key="time", joint_attr="X_pca", conditional_attr={"attr": "obs", "key": "time"})
        problem = problem.solve(**neuraldual_args_1)

        solver = problem.solver.solver
        assert solver.cond_dim > 0
        for arg, val in neuraldual_solver_args.items():
            assert hasattr(solver, val)
            el = getattr(solver, val)[0] if isinstance(getattr(solver, val), tuple) else getattr(solver, val)
            assert el == neuraldual_args_1[arg]

    def test_pass_custom_optimizers(self, adata_time: ad.AnnData):
        problem = ConditionalNeuralProblem(adata=adata_time)
        adata_time = adata_time[adata_time.obs["time"].isin((0, 1))]
        problem = problem.prepare(key="time", joint_attr="X_pca", conditional_attr={"attr": "obs", "key": "time"})
        custom_opt_f = optax.adagrad(1e-4)
        custom_opt_g = optax.adagrad(1e-3)

        problem = problem.solve(iterations=2, optimizer_f=custom_opt_f, optimizer_g=custom_opt_g)

    # def test_learning_rescaling_factors(self, adata_time: ad.AnnData):
    #     hidden_dim = 10
    #     problem = ConditionalNeuralProblem(adata=adata_time)
    #     mlp_eta = MLP_marginal(hidden_dim)
    #     mlp_xi = MLP_marginal(hidden_dim)
    #     adata_time = adata_time[adata_time.obs["time"].isin((0, 1))]
    #     problem = problem.prepare(key="time", joint_attr="X_pca", conditional_attr={"attr": "obs", "key": "time"})
    #     problem = problem.solve(mlp_eta=mlp_eta, mlp_xi=mlp_xi, **neuraldual_args_2)
    #     assert isinstance(problem.solution, BaseSolverOutput)
    #     assert isinstance(problem.solution, NeuralDualOutput)

    #     array = np.asarray(adata_time.obsm["X_pca"].copy())
    #     cond1 = jnp.ones((array.shape[0],))
    #     cond2 = jnp.zeros((array.shape[0],))
    #     learnt_eta_1 = problem.solution.evaluate_a(cond1, array)
    #     learnt_xi_1 = problem.solution.evaluate_b(cond1, array)
    #     learnt_eta_2 = problem.solution.evaluate_a(cond2, array)
    #     learnt_xi_2 = problem.solution.evaluate_b(cond2, array)
    #     assert learnt_eta_1.shape == (array.shape[0], 1)
    #     assert learnt_xi_1.shape == (array.shape[0], 1)
    #     assert learnt_eta_2.shape == (array.shape[0], 1)
    #     assert learnt_xi_2.shape == (array.shape[0], 1)
    #     assert np.sum(np.isnan(learnt_eta_1)) == 0
    #     assert np.sum(np.isnan(learnt_xi_1)) == 0
    #     assert np.sum(np.isnan(learnt_eta_2)) == 0
    #     assert np.sum(np.isnan(learnt_xi_2)) == 0

    #     np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, learnt_eta_1, learnt_eta_2)
    #     np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, learnt_xi_1, learnt_xi_2)
