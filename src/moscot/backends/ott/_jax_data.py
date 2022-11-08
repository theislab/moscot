from typing import List, Tuple, Optional
from dataclasses import dataclass

from ott.core.sinkhorn import sinkhorn
from ott.geometry.pointcloud import PointCloud
import jax
import jax.numpy as jnp


@dataclass
class JaxSampler:
    """Data sampler for Jax."""

    data_source: jnp.ndarray
    data_target: jnp.ndarray
    batch_size: int = 1024
    weighting: Optional[jnp.ndarray] = None
    tau_a: Optional[float] = None
    tau_b: Optional[float] = None
    epsilon: Optional[float] = None

    def __post_init__(self):
        self.length_source: int = len(self.data_source)
        self.length_target: int = len(self.data_target)
        if self.tau_a is not None or self.tau_b is not None:
            self.tau_a = 1.0 if self.tau_a is None else self.tau_a
            self.tau_b = 1.0 if self.tau_b is None else self.tau_b
            self.epsilon = 0.1 if self.epsilon is None else self.epsilon
            self.matching = True
            # always use uniform marginals for siknhorn
            self.ott_scaling: jnp.ndarray = jnp.ones(self.batch_size) / self.batch_size
        else:
            self.matching = False

        @jax.jit
        def _sample(key: jax.random.KeyArray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Jitted sample function."""
            key_source, key_target = jax.random.split(key, 2)
            if self.weighting is None:
                batch_source = self.data_source[
                    jax.random.randint(key_source, shape=[self.batch_size], minval=0, maxval=self.length_source)
                ]
            else:
                batch_source = jax.random.choice(key, self.data_source, shape=[self.batch_size], p=self.weighting)
            batch_target = self.data_target[
                jax.random.randint(key_target, shape=[self.batch_size], minval=0, maxval=self.length_target)
            ]
            return batch_source, batch_target

        @jax.jit
        def _matching_sample(
            key: jax.random.KeyArray,
            batch_source: jnp.ndarray,
            batch_target: jnp.ndarray,
            transition_matrix: jnp.ndarray,
        ) -> List[jnp.ndarray]:
            """Jitted sample function."""
            key_source, key_target, key_indices = jax.random.split(key, 3)
            if batch_source is None:
                if self.weighting is None:
                    batch_source = self.data_source[
                        jax.random.randint(key_source, shape=[self.batch_size], minval=0, maxval=self.length_source)
                    ]
                else:
                    batch_source = jax.random.choice(key, self.data_source, shape=[self.batch_size], p=self.weighting)
                batch_target = self.data_target[
                    jax.random.randint(key_target, shape=[self.batch_size], minval=0, maxval=self.length_target)
                ]
                # solve regularized ot between batch_source and batch_target
                geom = PointCloud(batch_source, batch_target, epsilon=self.epsilon, scale_cost="mean")
                out = sinkhorn(
                    geom,
                    a=self.ott_scaling,
                    b=self.ott_scaling,
                    tau_a=self.tau_a,
                    tau_b=self.tau_b,
                    jit=False,
                    max_iterations=1e7,
                )
                # get flattened log transition matrix because jax uses log probabilities
                transition_matrix = jnp.log(geom.transport_from_potentials(out.f, out.g).flatten())
            # sample from transition_matrix
            indeces = jax.random.categorical(key_indices, transition_matrix, shape=[self.batch_size])
            indeces_source = indeces // self.batch_size
            indeces_target = indeces % self.batch_size
            return [
                batch_source[indeces_source],
                batch_target[indeces_target],
                batch_source,
                batch_target,
                transition_matrix,
            ]

        self._sample = _sample
        self._matching_sample = _matching_sample

    def __call__(
        self, key: jax.random.KeyArray, full_dataset: bool = False, inner_iter: Optional[int] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Sample data."""
        if full_dataset:
            return self.data_source, self.data_target
        if self.matching:
            # resample and recompute transition matrix at every first inner iteration
            if inner_iter == 0:
                self.curr_source_batch: jnp.ndarray = None
                self.curr_target_batch: jnp.ndarray = None
                self.transition_matrix: jnp.ndarray = None
            (
                source,
                target,
                self.curr_source_batch,
                self.curr_target_batch,
                self.transition_matrix,
            ) = self._matching_sample(key, self.curr_source_batch, self.curr_target_batch, self.transition_matrix)
            return source, target
        return self._sample(key)
