from typing import Any, Optional

from ott.geometry.pointcloud import PointCloud
from ott.tools.sinkhorn_divergence import sinkhorn_divergence
import jax.numpy as jnp

from moscot._types import ArrayLike, ScaleCost_t
from moscot._logging import logger


def _compute_sinkhorn_divergence(
    point_cloud_1: ArrayLike,
    point_cloud_2: ArrayLike,
    a: Optional[ArrayLike] = None,
    b: Optional[ArrayLike] = None,
    epsilon: float = 10.0, #TODO(@MUCDK) enable once fixed on ott-jax
    tau_a: float = 1.0,
    tau_b: float = 1.0,
    scale_cost: ScaleCost_t = 1.0, #TODO(@MUCDK) enable once fixed on ott-jax
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
        # tau_a=tau_a,
        # tau_b=tau_b,
        # a=a,
        # b=b,
        # epsilon=epsilon,
        # scale_cost=scale_cost,
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
