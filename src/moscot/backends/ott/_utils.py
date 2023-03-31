from typing import Any, Optional

import jax.numpy as jnp
import scipy.sparse as sp
from ott.geometry.geometry import Geometry
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


def check_shapes(geom_x: Geometry, geom_y: Geometry, geom_xy: Geometry) -> None:
    n, m = geom_xy.shape
    n_, m_ = geom_x.shape[0], geom_y.shape[0]
    if n != n_:
        raise ValueError(f"Expected the first geometry to have `{n}` points, found `{n_}`.")
    if m != m_:
        raise ValueError(f"Expected the second geometry to have `{m}` points, found `{m_}`.")


def alpha_to_fused_penalty(alpha: float) -> float:
    if not (0 < alpha <= 1):
        raise ValueError(f"Expected `alpha` to be in interval `(0, 1]`, found `{alpha}`.")
    return (1 - alpha) / alpha


def ensure_2d(arr: ArrayLike, *, reshape: bool = False) -> Optional[jnp.ndarray]:  # type: ignore[name-defined]
    if sp.issparse(arr):
        arr = arr.A  # type: ignore[attr-defined]
    arr = jnp.asarray(arr)
    if reshape and arr.ndim == 1:
        return jnp.reshape(arr, (-1, 1))
    if arr.ndim != 2:
        raise ValueError(f"Expected array to have 2 dimensions, found `{arr.ndim}`.")
    return arr
