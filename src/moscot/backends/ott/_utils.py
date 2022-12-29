from typing import Any, Tuple, Optional
from functools import partial

from ott.geometry.pointcloud import PointCloud
from ott.tools.sinkhorn_divergence import sinkhorn_divergence
import jax
import jax.numpy as jnp

from moscot._types import ArrayLike, ScaleCost_t
from moscot._logging import logger


def _compute_sinkhorn_divergence(
    point_cloud_1: ArrayLike,
    point_cloud_2: ArrayLike,
    a: Optional[ArrayLike] = None,
    b: Optional[ArrayLike] = None,
    epsilon: float = 10.0, 
    tau_a: float = 1.0,
    tau_b: float = 1.0,
    scale_cost: ScaleCost_t = 1.0, 
    **kwargs: Any,
) -> float:
    point_cloud_1 = jnp.asarray(point_cloud_1)
    point_cloud_2 = jnp.asarray(point_cloud_2)
    a = None if a is None else jnp.asarray(a)
    b = None if b is None else jnp.asarray(b)

    if tau_a != 1.0:
        tau_a = 1.0
        logger.info(f"Setting `tau_a` from {tau_a} to 1.0 until fixed on OTT side.")
    if tau_b != 1.0:
        tau_b = 1.0
        logger.info(f"Setting `tau_b` from {tau_b} to 1.0 until fixed on OTT side.")

    output = sinkhorn_divergence(
        PointCloud,
        x=point_cloud_1,
        y=point_cloud_2,
        a=a,
        b=b,
        sinkhorn_kwargs = {"tau_a": tau_a, "tau_b": tau_b, "epsilon": epsilon},
        scale_cost=scale_cost,
        **kwargs,
    )
    xy_conv, xx_conv, *yy_conv = output.converged

    if not xy_conv:
        logger.warning("Solver did not converge in the `x/y` term.")
    if not xx_conv:
        logger.warning("Solver did not converge in the `x/x` term.")
    if len(yy_conv) and not yy_conv[0]:
        logger.warning("Solver did not converge in the `y/y` term.")

    return float(output.divergence)


class RunningAverageMeter:
    """Computes and stores the average value."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset the meter."""
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: float = 0.0

    def update(self, val: float, n: int = 1) -> None:
        """Update the meter."""
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@partial(jax.jit, static_argnames=["k"])
def get_nearest_neighbors(
    input_batch: jnp.ndarray, target: jnp.ndarray, k: int = 30
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Get the k nearest neighbors of the input batch in the target."""
    k = min(input_batch.shape[0], k)
    pairwise_euclidean_distances = -1 * jnp.sqrt(jnp.sum((input_batch - target) ** 2, axis=-1))
    distances, indices = jax.lax.top_k(pairwise_euclidean_distances, k=k)
    return distances, indices
