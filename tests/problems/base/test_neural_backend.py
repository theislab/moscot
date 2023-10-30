import pytest
from flax.training import train_state

import jax
import jax.numpy as jnp
import numpy as np
from ott.problems.linear.potentials import DualPotentials

from moscot.backends.ott._jax_data import JaxSampler
from moscot.backends.ott._neuraldual import OTTNeuralDualSolver
from moscot.backends.ott.nets._icnn import ICNN
from moscot.backends.ott.nets._nets import MLP_marginal


class TestJaxSampler:
    @pytest.mark.parametrize("full_dataset", [True, False])
    def test_jaxsampler_no_condition(self, sampler_no_conditions: JaxSampler, full_dataset: bool):
        data_dim = sampler_no_conditions.distributions[0].shape[1]
        bs = sampler_no_conditions.batch_size if not full_dataset else len(sampler_no_conditions.distributions[0])
        # this only works as all conditions have same number of observations

        out = sampler_no_conditions(jax.random.PRNGKey(0), (0, 1), "source", full_dataset=full_dataset)
        assert isinstance(out, tuple)
        assert len(out) == 2
        assert isinstance(out[0], jnp.ndarray)
        assert out[0].shape == (bs, data_dim)
        assert out[1] is None

        out = sampler_no_conditions(jax.random.PRNGKey(0), (0, 1), "target", full_dataset=full_dataset)
        assert isinstance(out, jnp.ndarray)
        assert out.shape == (bs, data_dim)

        out = sampler_no_conditions(jax.random.PRNGKey(0), (0, 1), "both", full_dataset=full_dataset)
        assert isinstance(out, tuple)
        assert len(out) == 3
        assert isinstance(out[0], jnp.ndarray)
        assert out[0].shape == (bs, data_dim)
        assert out[1] is None
        assert isinstance(out[2], jnp.ndarray)
        assert out[2].shape == (bs, data_dim)

        out_marginals = sampler_no_conditions.compute_unbalanced_marginals(out[0], out[2])
        assert isinstance(out_marginals, tuple)
        assert len(out_marginals) == 2
        assert isinstance(out_marginals[0], jnp.ndarray)
        assert out_marginals[0].shape == (bs,)
        assert isinstance(out_marginals[1], jnp.ndarray)
        assert out_marginals[1].shape == (bs,)

        out_resampled = sampler_no_conditions.unbalanced_resample(jax.random.PRNGKey(0), (out[0],), out_marginals[0])
        assert isinstance(out_resampled, tuple)
        assert len(out_resampled) == 1
        assert isinstance(out_resampled[0], jnp.ndarray)
        assert out_resampled[0].shape == (bs, data_dim)

    @pytest.mark.parametrize("full_dataset", [True, False])
    def test_jaxsampler_with_condition(self, sampler_with_conditions: JaxSampler, full_dataset: bool):
        data_dim = sampler_with_conditions.distributions[0].shape[1]
        cond_dim = sampler_with_conditions.conditions[0].shape[1]
        bs = sampler_with_conditions.batch_size if not full_dataset else len(sampler_with_conditions.distributions[0])
        # this only works as all conditions have same number of observations
        out = sampler_with_conditions(jax.random.PRNGKey(0), (0, 1), "source", full_dataset=full_dataset)
        assert isinstance(out, tuple)
        assert len(out) == 2
        assert isinstance(out[0], jnp.ndarray)
        assert out[0].shape == (bs, data_dim)
        assert isinstance(out[1], jnp.ndarray)
        assert out[1].shape == (bs, cond_dim)
        assert len(out[0]) == len(out[1])

        out = sampler_with_conditions(jax.random.PRNGKey(0), (0, 1), "target", full_dataset=full_dataset)
        assert isinstance(out, jnp.ndarray)
        assert out.ndim == 2

        out = sampler_with_conditions(jax.random.PRNGKey(0), (0, 1), "both", full_dataset=full_dataset)
        assert isinstance(out, tuple)
        assert len(out) == 3
        assert isinstance(out[0], jnp.ndarray)
        assert out[0].shape == (bs, data_dim)
        assert isinstance(out[1], jnp.ndarray)
        assert out[1].shape == (bs, cond_dim)
        assert isinstance(out[2], jnp.ndarray)
        assert out[2].shape == (bs, data_dim)

        out_marginals = sampler_with_conditions.compute_unbalanced_marginals(out[0], out[2])
        assert isinstance(out_marginals, tuple)
        assert len(out_marginals) == 2
        assert isinstance(out_marginals[0], jnp.ndarray)
        assert out_marginals[0].shape == (bs,)
        assert isinstance(out_marginals[1], jnp.ndarray)
        assert out_marginals[1].shape == (bs,)

        out_resampled = sampler_with_conditions.unbalanced_resample(jax.random.PRNGKey(0), (out[0],), out_marginals[0])
        assert isinstance(out_resampled, tuple)
        assert len(out_resampled) == 1
        assert isinstance(out_resampled[0], jnp.ndarray)
        assert out_resampled[0].shape == (bs, data_dim)


class TestICNN:
    def test_icnn(self):
        # test adapted from OTT-JAX
        input_dim = 10
        n_samples = 100
        model = ICNN([3, 3, 3], input_dim=input_dim, cond_dim=0, pos_weights=True)

        rng1 = jax.random.PRNGKey(1)
        rng2 = jax.random.PRNGKey(2)
        params = model.init(rng1, jnp.ones(input_dim))["params"]

        x = jax.random.normal(rng1, (n_samples, input_dim)) * 0.1
        y = jax.random.normal(rng2, (n_samples, input_dim))
        out_x = model.apply({"params": params}, x)
        out_y = model.apply({"params": params}, y)

        out = []
        for t in jnp.linspace(0, 1):
            out_xy = model.apply({"params": params}, t * x + (1 - t) * y)
            out.append((t * out_x + (1 - t) * out_y) - out_xy)

        np.testing.assert_array_equal(jnp.asarray(out) >= 0, True)

    def test_cond_icnn(self):
        input_dim = 11
        n_samples = 100
        cond_dim = 5
        model = ICNN([3, 3, 3], input_dim=input_dim, cond_dim=cond_dim)

        rng1 = jax.random.PRNGKey(1)
        rng2 = jax.random.PRNGKey(2)
        params = model.init(rng1, jnp.ones(input_dim), jnp.ones(cond_dim))["params"]

        x = jax.random.normal(rng1, (n_samples, input_dim))
        cond = jax.random.normal(rng2, (n_samples, cond_dim))

        out = model.apply({"params": params}, x, cond)
        assert isinstance(out, jnp.ndarray)


class TestMLPMarginals:
    def test_mlp_marginal(self):
        input_dim = 10
        n_samples = 100
        model = MLP_marginal(32)

        params = model.init(jax.random.PRNGKey(0), jnp.ones(input_dim))["params"]
        x = jax.random.normal(jax.random.PRNGKey(0), (n_samples, input_dim))

        out_x = model.apply({"params": params}, x)
        assert isinstance(out_x, jnp.ndarray)
        assert out_x.shape == (n_samples, 1)


class TestOTTNeuralDualSolver:
    def test_ott_neural_dual_solver_balanced_unconditional(self, sampler_no_conditions: JaxSampler):
        input_dim = sampler_no_conditions.distributions[0].shape[1]
        solver = OTTNeuralDualSolver(input_dim=input_dim, cond_dim=0, batch_size=32, iterations=3)

        out = solver(sampler_no_conditions, sampler_no_conditions)
        assert isinstance(out, tuple)
        assert len(out) == 3
        assert isinstance(out[0], DualPotentials)
        assert isinstance(out[1], OTTNeuralDualSolver)
        assert isinstance(out[2], dict)

        assert solver.cond_dim == 0
        assert solver.tau_a == 1.0
        assert solver.tau_b == 1.0
        assert solver.state_eta is None
        assert solver.state_xi is None
        assert solver.opt_eta is None
        assert solver.opt_xi is None

    def test_ott_neural_dual_solver_balanced_conditional(self, sampler_with_conditions: JaxSampler):
        input_dim = sampler_with_conditions.distributions[0].shape[1]
        cond_dim = sampler_with_conditions.conditions[0].shape[1]
        solver = OTTNeuralDualSolver(input_dim=input_dim, cond_dim=cond_dim, batch_size=32, iterations=3)

        out = solver(sampler_with_conditions, sampler_with_conditions)
        assert isinstance(out, tuple)
        assert len(out) == 3
        assert isinstance(out[0], DualPotentials)
        assert isinstance(out[1], OTTNeuralDualSolver)
        assert isinstance(out[2], dict)

        assert solver.cond_dim == 1
        assert solver.tau_a == 1.0
        assert solver.tau_b == 1.0
        assert solver.state_eta is None
        assert solver.state_xi is None
        assert solver.opt_eta is None
        assert solver.opt_xi is None

    def test_ott_neural_dual_solver_unbalanced(self, sampler_no_conditions: JaxSampler):
        input_dim = sampler_no_conditions.distributions[0].shape[1]
        mlp_eta = MLP_marginal(32)
        mlp_xi = MLP_marginal(32)
        solver = OTTNeuralDualSolver(
            input_dim, cond_dim=0, batch_size=32, iterations=3, tau_a=0.5, tau_b=0.6, mlp_eta=mlp_eta, mlp_xi=mlp_xi
        )

        out = solver(sampler_no_conditions, sampler_no_conditions)
        assert isinstance(out, tuple)
        assert len(out) == 3
        assert isinstance(out[0], DualPotentials)
        assert isinstance(out[1], OTTNeuralDualSolver)
        assert isinstance(out[2], dict)

        assert solver.cond_dim == 0
        assert solver.tau_a == 0.5
        assert solver.tau_b == 0.6
        assert isinstance(solver.state_eta, train_state.TrainState)
        assert isinstance(solver.state_xi, train_state.TrainState)
        assert solver.opt_eta is not None
        assert solver.opt_xi is not None

    def test_ott_neural_dual_solver_unbalanced_conditional(self, sampler_with_conditions: JaxSampler):
        input_dim = sampler_with_conditions.distributions[0].shape[1]
        cond_dim = sampler_with_conditions.conditions[0].shape[1]
        mlp_eta = MLP_marginal(32)
        mlp_xi = MLP_marginal(32)
        solver = OTTNeuralDualSolver(
            input_dim,
            cond_dim=cond_dim,
            batch_size=32,
            iterations=3,
            tau_a=0.5,
            tau_b=0.6,
            mlp_eta=mlp_eta,
            mlp_xi=mlp_xi,
        )

        out = solver(sampler_with_conditions, sampler_with_conditions)
        assert isinstance(out, tuple)
        assert len(out) == 3
        assert isinstance(out[0], DualPotentials)
        assert isinstance(out[1], OTTNeuralDualSolver)
        assert isinstance(out[2], dict)

        assert solver.cond_dim == 1
        assert solver.tau_a == 0.5
        assert solver.tau_b == 0.6
        assert isinstance(solver.state_eta, train_state.TrainState)
        assert isinstance(solver.state_xi, train_state.TrainState)
        assert solver.opt_eta is not None
        assert solver.opt_xi is not None
