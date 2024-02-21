import optax
import pytest

import jax.numpy as jnp
import numpy as np
from ott.geometry import costs

import anndata as ad

from moscot.backends.ott.output import OTTNeuralOutput
from moscot.base.output import BaseSolverOutput
from moscot.base.problems import CondOTProblem
from moscot.problems.generic import (
    GENOTLinProblem, GENOTQuadProblem  # type: ignore[attr-defined]
)
from moscot.utils.tagged_array import DistributionCollection, DistributionContainer
from tests._utils import ATOL, RTOL
from tests.problems.conftest import (
    neurallin_cond_args_1,
    neurallin_cond_args_2,
)

@pytest.fixture(scope="session")
def adata_quad(adata_time):
    adata = adata_time.copy()
    adata.obsm['Y_pca'] = adata.obsm['X_pca'][:, :-2]
    return adata


class TestGENOTLinProblem:
    @pytest.mark.fast()
    def test_prepare(self, adata_time: ad.AnnData):
        problem = GENOTLinProblem(adata=adata_time)
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
        problem = GENOTLinProblem(adata=adata_time)
        problem = problem.prepare(key="time", joint_attr="X_pca", conditional_attr={"attr": "obs", "key": "time"})
        problem = problem.solve(train_size=train_size, **neurallin_cond_args_1)
        assert isinstance(problem.solution, BaseSolverOutput)

    def test_solve_unbalanced_with_baseline(self, adata_time: ad.AnnData):
        problem = GENOTLinProblem(adata=adata_time)
        problem = problem.prepare(key="time", joint_attr="X_pca", conditional_attr={"attr": "obs", "key": "time"})
        problem = problem.solve(**neurallin_cond_args_2)
        assert isinstance(problem.solution, BaseSolverOutput)

    def test_reproducibility(self, adata_time: ad.AnnData):
        cond_zero_mask = np.array(adata_time.obs["time"] == 0)
        pc_tzero = adata_time[cond_zero_mask].obsm["X_pca"]
        problem_one = GENOTLinProblem(adata=adata_time)
        problem_one = problem_one.prepare(
            key="time", joint_attr="X_pca", conditional_attr={"attr": "obs", "key": "time"}
        )
        problem_one = problem_one.solve(**neurallin_cond_args_1)

        problem_two = GENOTLinProblem(adata=adata_time)
        problem_two = problem_one.prepare("time", joint_attr="X_pca", conditional_attr={"attr": "obs", "key": "time"})
        problem_two = problem_one.solve(**neurallin_cond_args_1)
        assert np.allclose(
            problem_one.solution.push(pc_tzero, cond=np.zeros((cond_zero_mask.sum(),1))),
            problem_two.solution.push(pc_tzero, cond=np.zeros((cond_zero_mask.sum(),1))),
            rtol=RTOL,
            atol=ATOL,
        )

    # def test_pass_arguments(self, adata_time: ad.AnnData): # TODO(ilan-gold) implement this once the OTT PR is settled
    #     problem = GENOTLinProblem(adata=adata_time)
    #     adata_time = adata_time[adata_time.obs["time"].isin((0, 1))]
    #     problem = problem.prepare(key="time", joint_attr="X_pca", conditional_attr={"attr": "obs", "key": "time"})
    #     problem = problem.solve(**neurallin_cond_args_1)

    #     solver = problem.solver._solver
    #     for arg, val in neurallin_cond_args_1.items():
    #         assert hasattr(solver, val)
    #         el = getattr(solver, val)[0] if isinstance(getattr(solver, val), tuple) else getattr(solver, val)
    #         assert el == neurallin_cond_args_1[arg]

    def test_pass_custom_optimizers(self, adata_time: ad.AnnData):
        problem = GENOTLinProblem(adata=adata_time)
        adata_time = adata_time[adata_time.obs["time"].isin((0, 1))]
        problem = problem.prepare(key="time", joint_attr="X_pca", conditional_attr={"attr": "obs", "key": "time"})
        custom_opt = optax.adagrad(1e-4)

        problem = problem.solve(iterations=2, optimizer=custom_opt)

    def test_learning_rescaling_factors(self, adata_time: ad.AnnData):
        problem = GENOTLinProblem(adata=adata_time)
        adata_time = adata_time[adata_time.obs["time"].isin((0, 1))]
        problem = problem.prepare(key="time", joint_attr="X_pca", conditional_attr={"attr": "obs", "key": "time"})
        problem = problem.solve(**neurallin_cond_args_2)
        assert isinstance(problem.solution, BaseSolverOutput)

        array = np.asarray(adata_time.obsm["X_pca"].copy())
        cond1 = jnp.ones((array.shape[0],1))
        cond2 = jnp.zeros((array.shape[0],1))
        learnt_eta_1 = problem.solver.solver.unbalancedness_handler.evaluate_eta(array, cond1) # TODO(ilan-gold): sould this be wrapped?
        learnt_xi_1 = problem.solver.solver.unbalancedness_handler.evaluate_xi(array, cond1)
        learnt_eta_2 = problem.solver.solver.unbalancedness_handler.evaluate_eta(array, cond2)
        learnt_xi_2 = problem.solver.solver.unbalancedness_handler.evaluate_xi(array, cond2)
        assert learnt_eta_1.shape == (array.shape[0], 4)
        assert learnt_xi_1.shape == (array.shape[0], 4)
        assert learnt_eta_2.shape == (array.shape[0], 4)
        assert learnt_xi_2.shape == (array.shape[0], 4)
        assert np.sum(np.isnan(learnt_eta_1)) == 0
        assert np.sum(np.isnan(learnt_xi_1)) == 0
        assert np.sum(np.isnan(learnt_eta_2)) == 0
        assert np.sum(np.isnan(learnt_xi_2)) == 0

        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, learnt_eta_1, learnt_eta_2)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, learnt_xi_1, learnt_xi_2)

class TestGENOTQuadProblem:
    @pytest.mark.fast()
    def test_prepare(self, adata_quad: ad.AnnData):
        problem = GENOTQuadProblem(adata=adata_quad)
        problem = problem.prepare(key="time", attrs=["X_pca", "Y_pca"], conditional_attr={"attr": "obs", "key": "time"})
        assert isinstance(problem, CondOTProblem)
        assert isinstance(problem.distributions, DistributionCollection)
        assert list(problem.distributions.keys()) == [0, 1, 2]

        container = problem.distributions[0]
        n_obs_0 = adata_quad[adata_quad.obs["time"] == 0].n_obs
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
        problem = GENOTQuadProblem(adata=adata_time)
        problem = problem.prepare(key="time", joint_attr="X_pca", conditional_attr={"attr": "obs", "key": "time"})
        problem = problem.solve(train_size=train_size, **neurallin_cond_args_1)
        assert isinstance(problem.solution, BaseSolverOutput)

    def test_solve_unbalanced_with_baseline(self, adata_time: ad.AnnData):
        problem = GENOTQuadProblem(adata=adata_time)
        problem = problem.prepare(key="time", joint_attr="X_pca", conditional_attr={"attr": "obs", "key": "time"})
        problem = problem.solve(**neurallin_cond_args_2)
        assert isinstance(problem.solution, BaseSolverOutput)

    def test_reproducibility(self, adata_time: ad.AnnData):
        cond_zero_mask = np.array(adata_time.obs["time"] == 0)
        pc_tzero = adata_time[cond_zero_mask].obsm["X_pca"]
        problem_one = GENOTQuadProblem(adata=adata_time)
        problem_one = problem_one.prepare(
            key="time", joint_attr="X_pca", conditional_attr={"attr": "obs", "key": "time"}
        )
        problem_one = problem_one.solve(**neurallin_cond_args_1)

        problem_two = GENOTQuadProblem(adata=adata_time)
        problem_two = problem_one.prepare("time", joint_attr="X_pca", conditional_attr={"attr": "obs", "key": "time"})
        problem_two = problem_one.solve(**neurallin_cond_args_1)
        assert np.allclose(
            problem_one.solution.push(pc_tzero, cond=np.zeros((cond_zero_mask.sum(),1))),
            problem_two.solution.push(pc_tzero, cond=np.zeros((cond_zero_mask.sum(),1))),
            rtol=RTOL,
            atol=ATOL,
        )

    # def test_pass_arguments(self, adata_time: ad.AnnData): # TODO(ilan-gold) implement this once the OTT PR is settled
    #     problem = GENOTQuadProblem(adata=adata_time)
    #     adata_time = adata_time[adata_time.obs["time"].isin((0, 1))]
    #     problem = problem.prepare(key="time", joint_attr="X_pca", conditional_attr={"attr": "obs", "key": "time"})
    #     problem = problem.solve(**neurallin_cond_args_1)

    #     solver = problem.solver._solver
    #     for arg, val in neurallin_cond_args_1.items():
    #         assert hasattr(solver, val)
    #         el = getattr(solver, val)[0] if isinstance(getattr(solver, val), tuple) else getattr(solver, val)
    #         assert el == neurallin_cond_args_1[arg]

    def test_pass_custom_optimizers(self, adata_time: ad.AnnData):
        problem = GENOTQuadProblem(adata=adata_time)
        adata_time = adata_time[adata_time.obs["time"].isin((0, 1))]
        problem = problem.prepare(key="time", joint_attr="X_pca", conditional_attr={"attr": "obs", "key": "time"})
        custom_opt = optax.adagrad(1e-4)

        problem = problem.solve(iterations=2, optimizer=custom_opt)

    def test_learning_rescaling_factors(self, adata_time: ad.AnnData):
        problem = GENOTQuadProblem(adata=adata_time)
        adata_time = adata_time[adata_time.obs["time"].isin((0, 1))]
        problem = problem.prepare(key="time", joint_attr="X_pca", conditional_attr={"attr": "obs", "key": "time"})
        problem = problem.solve(**neurallin_cond_args_2)
        assert isinstance(problem.solution, BaseSolverOutput)

        array = np.asarray(adata_time.obsm["X_pca"].copy())
        cond1 = jnp.ones((array.shape[0],1))
        cond2 = jnp.zeros((array.shape[0],1))
        learnt_eta_1 = problem.solver.solver.unbalancedness_handler.evaluate_eta(array, cond1) # TODO(ilan-gold): sould this be wrapped?
        learnt_xi_1 = problem.solver.solver.unbalancedness_handler.evaluate_xi(array, cond1)
        learnt_eta_2 = problem.solver.solver.unbalancedness_handler.evaluate_eta(array, cond2)
        learnt_xi_2 = problem.solver.solver.unbalancedness_handler.evaluate_xi(array, cond2)
        assert learnt_eta_1.shape == (array.shape[0], 4)
        assert learnt_xi_1.shape == (array.shape[0], 4)
        assert learnt_eta_2.shape == (array.shape[0], 4)
        assert learnt_xi_2.shape == (array.shape[0], 4)
        assert np.sum(np.isnan(learnt_eta_1)) == 0
        assert np.sum(np.isnan(learnt_xi_1)) == 0
        assert np.sum(np.isnan(learnt_eta_2)) == 0
        assert np.sum(np.isnan(learnt_xi_2)) == 0

        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, learnt_eta_1, learnt_eta_2)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, learnt_xi_1, learnt_xi_2)
