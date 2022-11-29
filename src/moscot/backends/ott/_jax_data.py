from typing import Tuple, Optional

from ott.core.sinkhorn import sinkhorn
from ott.geometry.pointcloud import PointCloud
import jax
import jax.numpy as jnp


class JaxSampler:
    """Data sampler for Jax."""

    def __init__(
        self,
        data_source: jnp.ndarray,
        data_target: jnp.ndarray,
        a: Optional[jnp.ndarray] = None,
        b: Optional[jnp.ndarray] = None,
        batch_size: int = 1024,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        epsilon: float = 0.1,
    ):
        """Initialize data sampler."""
        self.data_source = data_source
        self.data_target = data_target
        self.a = a
        self.b = b
        self.batch_size = batch_size
        self.tau_a = tau_a
        self.tau_b = tau_b
        self.epsilon = epsilon

        @jax.jit
        def _sample_source(key: jax.random.KeyArray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Jitted sample function."""
            return jax.random.choice(key, self.data_source, shape=[self.batch_size], p=self.a)

        @jax.jit
        def _sample_target(key: jax.random.KeyArray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Jitted sample function."""
            return jax.random.choice(key, self.data_target, shape=[self.batch_size], p=self.b)

        @jax.jit
        def _compute_unbalanced_marginals(
            batch_source: jnp.ndarray,
            batch_target: jnp.ndarray,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Jitted function to compute the source and target marginals for a batch."""
            geom = PointCloud(batch_source, batch_target, epsilon=self.epsilon, scale_cost="mean")
            out = sinkhorn(
                geom,
                tau_a=self.tau_a,
                tau_b=self.tau_b,
                jit=False,
                max_iterations=1e7,
            )
            # get flattened log transition matrix
            transition_matrix = geom.transport_from_potentials(out.f, out.g)
            # jax categorical uses log probabilities
            log_marginals_source = jnp.log(jnp.sum(transition_matrix, axis=1))
            log_marginals_target = jnp.log(jnp.sum(transition_matrix, axis=0))
            return log_marginals_source, log_marginals_target

        @jax.jit
        def _unbalanced_resample(
            key: jax.random.KeyArray,
            batch: jnp.ndarray,
            log_marginals: jnp.ndarray,
        ) -> jnp.ndarray:
            """Resample a batch based upon log marginals."""
            # sample from marginals
            indices = jax.random.categorical(key, log_marginals, shape=[self.batch_size])
            return batch[indices]

        self._sample_source = _sample_source
        self._sample_target = _sample_target
        self.compute_unbalanced_marginals = _compute_unbalanced_marginals
        self.unbalanced_resample = _unbalanced_resample

    def __call__(
        self,
        key: jax.random.KeyArray,
        source: bool,
        full_dataset: bool = False,
    ) -> jnp.ndarray:
        """Sample data."""
        if full_dataset:
            if source:
                return self.data_source
            return self.data_target
        if source:
            return self._sample_source(key)
        return self._sample_target(key)
