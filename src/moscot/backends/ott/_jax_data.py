from typing import Any, Dict, List, Tuple, Optional

from ott.geometry.pointcloud import PointCloud
from ott.solvers.linear.sinkhorn import sinkhorn
import jax
import numpy as np
import jax.numpy as jnp


class JaxSampler:
    """Data sampler for Jax."""

    def __init__(
        self,
        distributions: List[jnp.ndarray],
        policies: List[Tuple[Any, Any]],
        a: Optional[jnp.ndarray] = None,
        b: Optional[jnp.ndarray] = None,
        sample2idx: Optional[Dict[int, Any]] = None,
        batch_size: int = 1024,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        epsilon: float = 0.1,
    ):
        """Initialize data sampler."""
        self.distributions = distributions
        self.policies = policies
        self.a = a
        self.b = b
        self.batch_size = batch_size
        self.tau_a = tau_a
        self.tau_b = tau_b
        self.epsilon = epsilon
        if sample2idx is None:
            if len(policies) > 1:
                raise ValueError("If `policies` contains more than 1 value, `sample2idx` is required.")
            sample2idx = {self.policies[0][0]:0, self.policies[0][1]:1}
        self.sample2idx = sample2idx

        def _sample_source(key: jax.random.KeyArray, s: Any, distributions) -> jnp.ndarray:
            """Jitted sample function."""
            print("s is ",s)
            return jax.random.choice(key, distributions[self.sample2idx[s]], shape=[self.batch_size], p=self.a)

        def _sample_target(key: jax.random.KeyArray,s : Any, distributions) -> jnp.ndarray:
            """Jitted sample function."""
            return jax.random.choice(key, distributions[self.sample2idx[s]], shape=[self.batch_size], p=self.b)

        def _sample(key: jax.random.KeyArray, distributions, policies) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Jitted sample function."""
            pair = policies[jax.random.choice(key, np.arange(len(policies)))]
            return self._sample_source(key, pair[0], distributions), self._sample_target(key, pair[1], distributions)

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

        def _unbalanced_resample(
            key: jax.random.KeyArray,
            batch: jnp.ndarray,
            log_marginals: jnp.ndarray,
        ) -> jnp.ndarray:
            """Resample a batch based upon log marginals."""
            # sample from marginals
            indices = jax.random.categorical(key, log_marginals, shape=[self.batch_size])
            return batch[indices]

        self._sample = _sample
        self._sample_source = _sample_source
        self._sample_target = _sample_target
        self.compute_unbalanced_marginals = _compute_unbalanced_marginals
        self.unbalanced_resample = _unbalanced_resample

    def __call__(
        self,
        key: jax.random.KeyArray,
        full_dataset: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Sample data."""
        print(self.sample2idx)
        if full_dataset:
            return np.vstack([self.distributions[self.sample2idx[s]] for s, _ in self.policies]), np.vstack(
                [self.distributions[self.sample2idx[s]] for _, s in self.policies]
            )
        return self._sample(key, self.distributions, self.policies)
