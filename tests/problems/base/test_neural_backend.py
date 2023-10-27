import jax
import jax.numpy as jnp
import numpy as np

from moscot.backends.ott._jax_data import JaxSampler
from moscot.backends.ott.nets._icnn import ICNN


class TestNeural:
    def test_jaxsampler_no_condition(self, sampler_no_conditions: JaxSampler):
        out = sampler_no_conditions(jax.random.PRNGKey(0), (0, 1), "source")
        assert isinstance(out, tuple)
        assert len(out) == 2
        assert isinstance(out[0], jnp.ndarray)
        assert out[1] is None

        out = sampler_no_conditions(jax.random.PRNGKey(0), (0, 1), "target")
        assert isinstance(out, jnp.ndarray)

    def test_jaxsampler_with_condition(self, sampler_with_conditions: JaxSampler):
        out = sampler_with_conditions(jax.random.PRNGKey(0), (0, 1), "source")
        assert isinstance(out, tuple)
        assert len(out) == 2
        assert isinstance(out[0], jnp.ndarray)
        assert isinstance(out[1], jnp.ndarray)
        assert len(out[0]) == len(out[1])

        out = sampler_with_conditions(jax.random.PRNGKey(0), (0, 1), "target")
        assert isinstance(out, jnp.ndarray)

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
