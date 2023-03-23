from typing import Any, Dict, List, Tuple, Union, Literal, Optional
from functools import partial

from ott.solvers.linear import sinkhorn
from ott.geometry.pointcloud import PointCloud
import jax
import jax.numpy as jnp


class JaxSampler:
    """Data sampler for Jax."""

    def __init__(
        self,
        distributions: List[jnp.ndarray],
        policy_pairs: List[Tuple[Any, Any]],
        a: List[jnp.ndarray] = None,
        b: List[jnp.ndarray] = None,
        sample_to_idx: Optional[Dict[int, Any]] = None,
        batch_size: int = 1024,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        epsilon: float = 0.1,
    ):
        """Initialize data sampler."""
        if not len(distributions) == len(a) == len(b):
            raise ValueError("Number of distributions, a, and b must be equal.")
        self._distributions = distributions
        self._policy_pairs = policy_pairs
        self._conditions = jnp.array(policy_pairs, dtype=float)
        if sample_to_idx is None:
            if len(self.policy_pairs) > 1:
                raise ValueError("If `policy_pairs` contains more than 1 value, `sample_to_idx` is required.")
            sample_to_idx = {self.policy_pairs[0][0]: 0, self.policy_pairs[0][1]: 1}
        self._sample_to_idx = sample_to_idx

        @partial(jax.jit, static_argnames=["index"])
        def _sample_source(key: jax.random.KeyArray, index: jnp.ndarray) -> jnp.ndarray:
            """Jitted sample function."""
            return jax.random.choice(key, self.distributions[index], shape=[batch_size], p=a[index])

        @partial(jax.jit, static_argnames=["index"])
        def _sample_target(key: jax.random.KeyArray, index: jnp.ndarray) -> jnp.ndarray:
            """Jitted sample function."""
            return jax.random.choice(key, self.distributions[index], shape=[batch_size], p=b[index])

        @jax.jit
        def _compute_unbalanced_marginals(
            batch_source: jnp.ndarray,
            batch_target: jnp.ndarray,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Jitted function to compute the source and target marginals for a batch."""
            geom = PointCloud(batch_source, batch_target, epsilon=epsilon, scale_cost="mean")
            out = sinkhorn.Sinkhorn(
                geom,
                tau_a=tau_a,
                tau_b=tau_b,
                jit=False,
                max_iterations=1e7,
            )
            # get log probabilities
            log_marginals_source = jnp.log(out.marginal(1))
            log_marginals_target = jnp.log(out.marginal(0))
            return log_marginals_source, log_marginals_target

        @jax.jit
        def _unbalanced_resample(
            key: jax.random.KeyArray,
            batch: jnp.ndarray,
            log_marginals: jnp.ndarray,
        ) -> jnp.ndarray:
            """Resample a batch based upon log marginals."""
            # sample from marginals
            indices = jax.random.categorical(key, log_marginals, shape=[batch_size])
            return batch[indices]

        def _sample_policy_pair(key: jax.random.KeyArray) -> Tuple[Tuple[Any, Any], Any]:
            """Sample a policy pair. If conditions are provided, return the policy pair and the conditions."""
            index = jax.random.randint(key, shape=[], minval=0, maxval=len(self.policy_pairs))
            policy_pair = self.policy_pairs[index]
            condition = self.conditions[index] if self.conditions is not None else None
            return policy_pair, condition

        self._sample_source = _sample_source
        self._sample_target = _sample_target
        self.sample_policy_pair = _sample_policy_pair
        self.compute_unbalanced_marginals = _compute_unbalanced_marginals
        self.unbalanced_resample = _unbalanced_resample

    def __call__(
        self,
        key: jax.random.KeyArray,
        policy_pair: Optional[Tuple[Any, Any]] = None,
        full_dataset: bool = False,
        sample: Literal["pair", "source", "target"] = "pair",
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Sample data."""
        if full_dataset:
            if sample == "source":
                return self.distributions[self.sample_to_idx[policy_pair[0]]]
            if sample == "target":
                return self.distributions[self.sample_to_idx[policy_pair[1]]]
            if sample == "pair":
                return (
                    self.distributions[self.sample_to_idx[policy_pair[0]]],
                    self.distributions[self.sample_to_idx[policy_pair[1]]],
                )
        if sample == "source":
            return self._sample_source(key, self.sample_to_idx[policy_pair[0]])
        if sample == "target":
            return self._sample_target(key, self.sample_to_idx[policy_pair[1]])
        return self._sample_source(key, self.sample_to_idx[policy_pair[0]]), self._sample_target(
            key, self.sample_to_idx[policy_pair[1]]
        )

    @property
    def distributions(self) -> List[jnp.ndarray]:
        """Return distributions."""
        return self._distributions

    @property
    def policy_pairs(self) -> List[Tuple[Any, Any]]:
        """Return policy pairs."""
        return self._policy_pairs

    @property
    def conditions(self) -> jnp.ndarray:
        """Return conditions."""
        return self._conditions

    @property
    def sample_to_idx(self) -> Dict[int, Any]:
        """Return sample to idx."""
        return self._sample_to_idx
