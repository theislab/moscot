from functools import partial
from types import MappingProxyType
from typing import Any, Dict, Hashable, List, Literal, Optional, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp
from ott.geometry.pointcloud import PointCloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

K = TypeVar("K", bound=Hashable)


class JaxSampler:
    """Data sampler for Jax."""

    def __init__(
        self,
        distributions: List[jnp.ndarray],
        policy_pairs: List[Tuple[Any, Any]],
        conditions: Optional[List[jnp.ndarray]],
        a: List[jnp.ndarray] = None,
        b: List[jnp.ndarray] = None,
        sample_to_idx: Dict[K, Any] = MappingProxyType({}),
        batch_size: int = 1024,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        epsilon: float = 0.1,
    ):
        """Initialize data sampler."""
        if not len(distributions) == len(a) == len(b):
            raise ValueError("Number of distributions, a, and b must be equal.")
        self._distributions = distributions
        self._conditions = conditions
        self._batch_size = batch_size
        self._policy_pairs = policy_pairs
        if not len(sample_to_idx):
            if len(self.policy_pairs) > 1:
                raise ValueError("If `policy_pairs` contains more than 1 value, `sample_to_idx` is required.")
            sample_to_idx = {self.policy_pairs[0][0]: 0, self.policy_pairs[0][1]: 1}
        self._sample_to_idx = sample_to_idx

        @partial(jax.jit, static_argnames=["index"])
        def _sample_source(key: jax.random.KeyArray, index: jnp.ndarray) -> Tuple[jnp.ndarray, None]:
            """Jitted sample function."""
            samples = jax.random.choice(key, self.distributions[index], shape=[batch_size], p=jnp.squeeze(a[index]))
            return jnp.asarray(samples), None

        @partial(jax.jit, static_argnames=["index"])
        def _sample_source_conditional(key: jax.random.KeyArray, index: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Jitted sample function."""
            samples = jax.random.choice(key, self.distributions[index], shape=[batch_size], p=jnp.squeeze(a[index]))
            conds = jax.random.choice(key, self.conditions[index], shape=[batch_size], p=jnp.squeeze(a[index]))  # type: ignore[index]  # noqa: E501
            return samples, conds

        @partial(jax.jit, static_argnames=["index"])
        def _sample_target(key: jax.random.KeyArray, index: jnp.ndarray) -> jnp.ndarray:
            """Jitted sample function."""
            return jax.random.choice(key, self.distributions[index], shape=[batch_size], p=jnp.squeeze(b[index]))

        @jax.jit
        def _compute_unbalanced_marginals(
            batch_source: jnp.ndarray,
            batch_target: jnp.ndarray,
            sinkhorn_kwargs: Dict[str, Any] = MappingProxyType({}),
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Jitted function to compute the source and target marginals for a batch."""
            geom = PointCloud(batch_source, batch_target, epsilon=epsilon, scale_cost="mean")
            out = sinkhorn.Sinkhorn(**sinkhorn_kwargs)(linear_problem.LinearProblem(geom, tau_a=tau_a, tau_b=tau_b))
            return out.matrix.sum(axis=1), out.matrix.sum(axis=0)

        @jax.jit
        def _unbalanced_resample(
            key: jax.random.KeyArray,
            batch: Tuple[jnp.ndarray, ...],
            marginals: jnp.ndarray,
        ) -> Tuple[jnp.ndarray, ...]:
            """Resample a batch based upon log marginals."""
            # sample from marginals
            indices = jax.random.choice(key, a=len(marginals), p=jnp.squeeze(marginals), shape=[len(marginals)])
            return tuple(b[indices] if b is not None else None for b in batch)

        def _sample_policy_pair(key: jax.random.KeyArray) -> Tuple[Any, Any]:
            """Sample a policy pair."""
            index = jax.random.randint(key, shape=[], minval=0, maxval=len(self.policy_pairs))
            return self.policy_pairs[index]

        self._sample_source = _sample_source if self.conditions is None else _sample_source_conditional
        self._sample_target = _sample_target
        self.sample_policy_pair = _sample_policy_pair
        self.compute_unbalanced_marginals = _compute_unbalanced_marginals
        self.unbalanced_resample = _unbalanced_resample

    def __call__(
        self,
        key: jax.random.KeyArray,
        policy_pair: Tuple[Any, Any],
        sample: Literal["source", "target", "both"] = "both",
        full_dataset: bool = False,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """Sample data. When sampling the source, the conditions are returned, too."""
        if full_dataset:
            if sample == "source":
                return jnp.asarray(
                    self.distributions[self.sample_to_idx[policy_pair[0]]]
                ), None if self.conditions is None else jnp.asarray(self.conditions[self.sample_to_idx[policy_pair[0]]])
            if sample == "target":
                return jnp.asarray(self.distributions[self.sample_to_idx[policy_pair[1]]])
            if sample == "both":
                return (
                    jnp.asarray(self.distributions[self.sample_to_idx[policy_pair[0]]]),
                    None
                    if self.conditions is None
                    else jnp.asarray(self.conditions[self.sample_to_idx[policy_pair[0]]]),
                    jnp.asarray(self.distributions[self.sample_to_idx[policy_pair[1]]]),
                )
            raise NotImplementedError(f"Sample type {sample} not implemented.")
        if sample == "source":
            return self._sample_source(key, self.sample_to_idx[policy_pair[0]])
        if sample == "target":
            return self._sample_target(key, self.sample_to_idx[policy_pair[1]])
        if sample == "both":
            return (
                *self._sample_source(key, self.sample_to_idx[policy_pair[0]]),
                self._sample_target(key, self.sample_to_idx[policy_pair[1]]),
            )
        raise NotImplementedError(f"Sample type {sample} not implemented.")

    @property
    def distributions(self) -> List[jnp.ndarray]:
        """Return distributions."""
        return self._distributions

    @property
    def policy_pairs(self) -> List[Tuple[Any, Any]]:
        """Return policy pairs."""
        return self._policy_pairs

    @property
    def conditions(self) -> Optional[jnp.ndarray]:
        """Return conditions."""
        return self._conditions

    @property
    def sample_to_idx(self) -> Dict[K, Any]:
        """Return sample to idx."""
        return self._sample_to_idx

    @property
    def batch_size(self) -> int:
        return self._batch_size
