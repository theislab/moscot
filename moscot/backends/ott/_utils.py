from typing import Any, Optional

from ott.geometry.pointcloud import PointCloud
from ott.tools.sinkhorn_divergence import sinkhorn_divergence

from moscot._types import ArrayLike, ScaleCost_t
from moscot._logging import logger


def _compute_sinkhorn_divergence(
    point_cloud_1: ArrayLike,
    point_cloud_2: ArrayLike,
    a: Optional[ArrayLike] = None,
    b: Optional[ArrayLike] = None,
    epsilon: float = 10,
    scale_cost: ScaleCost_t = 1,
    **kwargs: Any,
) -> float:
    output = sinkhorn_divergence(
        PointCloud, x=point_cloud_1, y=point_cloud_2, a=a, b=b, epsilon=epsilon, scale_cost=scale_cost, **kwargs
    )
    if not output.converged[0]:
        logger.warning("TODO: Solver not converged in x to y term.")
    if not output.converged[1]:
        logger.warning("TODO: Solver not converged in x to x term.")
    if len(output.converged) > 1 and not output.converged[2]:
        logger.warning("TODO: Solver not converged in y to y term.")
    return float(output.divergence)
