import inspect
from typing import Any, Callable, Dict, Optional

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
    # TODO(MUCDK): why is this so high?
    epsilon: float = 10.0,
    scale_cost: ScaleCost_t = 1.0,
    **kwargs: Any,
) -> float:
    point_cloud_1 = jnp.asarray(point_cloud_1)
    point_cloud_2 = jnp.asarray(point_cloud_2)
    a = None if a is None else jnp.asarray(a)
    b = None if b is None else jnp.asarray(b)

    output = sinkhorn_divergence(
        PointCloud, x=point_cloud_1, y=point_cloud_2, a=a, b=b, epsilon=epsilon, scale_cost=scale_cost, **kwargs
    )
    xy_conv, xx_conv, *yy_conv = output.converged

    if not xy_conv:
        logger.warning("Solver did not converge in the `x/y` term.")
    if not xx_conv:
        logger.warning("Solver did not converge in the `x/x` term.")
    if len(yy_conv) and not yy_conv[0]:
        logger.warning("Solver did not converge in the `y/y` term.")

    return float(output.divergence)


def _filter_kwargs(*funcs: Callable[..., Any], **kwargs: Any) -> Dict[str, Any]:
    res = {}
    for func in funcs:
        params = inspect.signature(func).parameters
        res.update({k: v for k, v in kwargs.items() if k in params})
    return res
