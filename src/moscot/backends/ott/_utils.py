import inspect
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from ott.geometry.pointcloud import PointCloud
from ott.tools.sinkhorn_divergence import sinkhorn_divergence

from moscot._logging import logger
from moscot._types import ArrayLike, ScaleCost_t


def _compute_sinkhorn_divergence(
    point_cloud_1: ArrayLike,
    point_cloud_2: ArrayLike,
    a: Optional[ArrayLike] = None,
    b: Optional[ArrayLike] = None,
    epsilon: Optional[float] = 1e-1,
    tau_a: float = 1.0,
    tau_b: float = 1.0,
    scale_cost: ScaleCost_t = 1.0,
    batch_size: Optional[int] = None,
    **kwargs: Any,
) -> float:
    point_cloud_1 = jnp.asarray(point_cloud_1)
    point_cloud_2 = jnp.asarray(point_cloud_2)
    a = None if a is None else jnp.asarray(a)
    b = None if b is None else jnp.asarray(b)

    output = sinkhorn_divergence(
        PointCloud,
        x=point_cloud_1,
        y=point_cloud_2,
        batch_size=batch_size,
        a=a,
        b=b,
        sinkhorn_kwargs={"tau_a": tau_a, "tau_b": tau_b},
        scale_cost=scale_cost,
        epsilon=epsilon,
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
    input_batch: jnp.ndarray, target: jnp.ndarray, k: int = 30  # type: ignore[name-defined]
) -> Tuple[jnp.ndarray, jnp.ndarray]:  # type: ignore[name-defined]
    """Get the k nearest neighbors of the input batch in the target."""
    if target.shape[0] < k:
        raise ValueError(f"k is {k}, but must be smaller or equal than {target.shape[0]}.")
    pairwise_euclidean_distances = jnp.sqrt(jnp.sum((input_batch - target) ** 2, axis=-1))
    negative_distances, indices = jax.lax.top_k(-1 * pairwise_euclidean_distances, k=k)
    return -1 * negative_distances, indices


def _filter_kwargs(*funcs: Callable[..., Any], **kwargs: Any) -> Dict[str, Any]:
    res = {}
    for func in funcs:
        params = inspect.signature(func).parameters
        res.update({k: v for k, v in kwargs.items() if k in params})
    return res
