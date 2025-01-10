from moscot.utils.subset_policy import SubsetPolicy
from moscot.neural.data._distribution_collection import DistributionCollection

import jax

from typing import Any, Dict, Iterator, List, Optional, Tuple


import jax
import jax.numpy as jnp
from typing import Any, Dict, Optional, List, Tuple, Iterator, Sequence

import functools


@functools.partial(jax.jit, static_argnums=(3,))
def _sample_indices(
    rng: jax.Array, idx_src: jnp.ndarray, idx_tgt: jnp.ndarray, batch_size: int
) -> Tuple[jax.Array, jnp.ndarray, jnp.ndarray]:
    """
    JIT-compiled function:
      - Splits RNG into rng_src/rng_tgt.
      - Samples with replacement from idx_src, idx_tgt.
      - Returns the updated rng, plus arrays of sample positions for src & tgt.
    """  # noqa: D205
    rng, rng_src, rng_tgt = jax.random.split(rng, 3)
    src_samples = jax.random.randint(rng_src, shape=(batch_size,), minval=0, maxval=idx_src.shape[0])
    tgt_samples = jax.random.randint(rng_tgt, shape=(batch_size,), minval=0, maxval=idx_tgt.shape[0])
    return rng, src_samples, tgt_samples


@jax.jit
def _gather_array(arr: jnp.ndarray, idxs: jnp.ndarray) -> jnp.ndarray:
    """
    JIT-compiled function to gather rows from arr at idxs.
    If arr is shape [N, ...], idxs is shape [K], result is [K, ...].
    """  # noqa: D205
    return jnp.take(arr, idxs, axis=0)


class PolicyDataLoader:
    """A data loader for handling subset policies and distribution collections.

    A data loader that:
      - Takes a SubsetPolicy (with a plan of edges).
      - Has a DistributionCollection mapping node -> DistributionContainer.
      - For each distribution container, we check that .xy, .xx, .conditions
        all share the same shape[0] if they're not None.
      - On each iteration:
        1) Randomly pick an edge (src_node, tgt_node) in Python.
        2) Use a small jitted function `_sample_indices` to sample from
           the node_indices (with replacement).
        3) Use a small jitted function `_gather_array` to gather data from
           .xy, .xx, .conditions.
        4) Build a final dictionary and yield it.
    """

    def __init__(
        self,
        rng: jax.Array,
        policy: SubsetPolicy[Any],
        distributions: DistributionCollection,
        batch_size: int = 128,
        plan: Optional[Sequence[Tuple[Any, Any]]] = None,
        src_prefix: str = "src",
        tgt_prefix: str = "tgt",
        src_renames: Optional[Dict[str, str]] = None,
        tgt_renames: Optional[Dict[str, str]] = None,
    ):

        self.policy = policy
        self.distributions = distributions
        self.rng = rng
        self.batch_size = batch_size
        self.edges = plan if plan is not None else self.policy.plan()
        self.src_prefix = src_prefix
        self.tgt_prefix = tgt_prefix
        self.src_renames = src_renames if src_renames is not None else {}
        self.tgt_renames = tgt_renames if tgt_renames is not None else {}

        # Precompute an index array for each node
        self.node_indices: Dict[Any, jnp.ndarray] = {}
        self._init_indices()

    def _init_indices(self) -> None:
        """Verify shape consistency within each DistributionContainer, store jnp.arange(...) as node_indices."""
        for node, container in self.distributions.items():
            # Gather shapes of non-None arrays
            shapes = []
            if container.xy is not None:
                shapes.append(container.xy.shape[0])
            if container.xx is not None:
                shapes.append(container.xx.shape[0])
            if container.conditions is not None:
                shapes.append(container.conditions.shape[0])

            # All must match
            if shapes and not all(s == shapes[0] for s in shapes):
                raise ValueError(f"Inconsistent shape for node {node}: {shapes}")

            if shapes:
                n = shapes[0]
                self.node_indices[node] = jnp.arange(n)

    def __iter__(self) -> Iterator[Dict[str, jnp.ndarray]]:
        """
        Infinite generator. Each iteration:
          1) Randomly pick an edge (src_node, tgt_node).
          2) Use _sample_indices(...) to get random sample positions from the node's data.
          3) Use _gather_array(...) to gather from .xy, .xx, .conditions.
          4) Build a dict and yield it.
        """
        while True:
            if not self.edges:
                break

            # (A) Pick a random edge in Python
            self.rng, rng_edge = jax.random.split(self.rng)
            i = jax.random.randint(rng_edge, shape=(), minval=0, maxval=len(self.edges))
            edge = self.edges[int(i)]
            src_node, tgt_node = edge

            # Skip if the node doesn't exist in distributions or indices
            if src_node not in self.distributions or tgt_node not in self.distributions:
                continue
            if src_node not in self.node_indices or tgt_node not in self.node_indices:
                continue

            src_container = self.distributions[src_node]
            tgt_container = self.distributions[tgt_node]
            idx_src = self.node_indices[src_node]
            idx_tgt = self.node_indices[tgt_node]

            # (B) Sample random positions with a small jitted function
            self.rng, src_samples, tgt_samples = _sample_indices(self.rng, idx_src, idx_tgt, self.batch_size)
            # Convert to actual indices
            src_idxs = jnp.take(idx_src, src_samples)
            tgt_idxs = jnp.take(idx_tgt, tgt_samples)

            # (C) Gather data from each relevant array
            batch_dict = {}

            src_candidates = [
                ("xy", src_container.xy),
                ("xx", src_container.xx),
                ("conditions", src_container.conditions),
            ]
            for key, arr in src_candidates:
                if arr is not None:
                    key_new = self.src_renames.get(key, key)
                    batch_dict[f"{self.src_prefix}_{key_new}"] = _gather_array(arr, src_idxs)

            tgt_candidates = [
                ("xy", tgt_container.xy),
                ("xx", tgt_container.xx),
                ("conditions", tgt_container.conditions),
            ]
            for key, arr in tgt_candidates:
                if arr is not None:
                    key_new = self.tgt_renames.get(key, key)
                    batch_dict[f"{self.tgt_prefix}_{key_new}"] = _gather_array(arr, tgt_idxs)
            if not batch_dict:
                continue

            yield batch_dict

    def __len__(self) -> int:
        """Optionally define a length if you like, e.g. len(edges)."""
        return len(self.edges)
