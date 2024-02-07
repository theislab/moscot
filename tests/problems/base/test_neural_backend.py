import pytest

import jax
import jax.numpy as jnp
import numpy as np

from moscot.backends.ott._jax_data import JaxSampler


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