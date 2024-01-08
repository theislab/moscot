from typing import Any, Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import scipy.sparse as sp
from ott.geometry import epsilon_scheduler, geodesic, geometry, pointcloud
from ott.tools import sinkhorn_divergence as sdiv

from moscot._logging import logger
from moscot._types import ArrayLike, ScaleCost_t
from moscot.utils.tagged_array import TaggedArray

Scale_t = Union[float, Literal["mean", "median", "max_cost", "max_norm", "max_bound"]]


__all__ = ["sinkhorn_divergence"]


def sinkhorn_divergence(
    point_cloud_1: ArrayLike,
    point_cloud_2: ArrayLike,
    a: Optional[ArrayLike] = None,
    b: Optional[ArrayLike] = None,
    epsilon: Union[float, epsilon_scheduler.Epsilon] = 1e-1,
    scale_cost: ScaleCost_t = 1.0,
    **kwargs: Any,
) -> float:
    point_cloud_1 = jnp.asarray(point_cloud_1)
    point_cloud_2 = jnp.asarray(point_cloud_2)
    a = None if a is None else jnp.asarray(a)
    b = None if b is None else jnp.asarray(b)

    output = sdiv.sinkhorn_divergence(
        pointcloud.PointCloud,
        x=point_cloud_1,
        y=point_cloud_2,
        a=a,
        b=b,
        epsilon=epsilon,
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


def check_shapes(geom_x: geometry.Geometry, geom_y: geometry.Geometry, geom_xy: geometry.Geometry) -> None:
    n, m = geom_xy.shape
    n_, m_ = geom_x.shape[0], geom_y.shape[0]
    if n != n_:
        raise ValueError(f"Expected the first geometry to have `{n}` points, found `{n_}`.")
    if m != m_:
        raise ValueError(f"Expected the second geometry to have `{m}` points, found `{m_}`.")


def alpha_to_fused_penalty(alpha: float) -> float:
    """Convert."""
    if not (0 < alpha <= 1):
        raise ValueError(f"Expected `alpha` to be in interval `(0, 1]`, found `{alpha}`.")
    return (1 - alpha) / alpha


def ensure_2d(arr: ArrayLike, *, reshape: bool = False) -> jax.Array:
    """Ensure that an array is 2-dimensional.

    Parameters
    ----------
    arr
        Array to check.
    reshape
        Allow reshaping 1-dimensional array to ``[n, 1]``.

    Returns
    -------
    2-dimensional :mod:`jax` array.
    """
    if sp.issparse(arr):
        arr = arr.A  # type: ignore[attr-defined]
    arr = jnp.asarray(arr)
    if reshape and arr.ndim == 1:
        return jnp.reshape(arr, (-1, 1))
    if arr.ndim != 2:
        raise ValueError(f"Expected array to have 2 dimensions, found `{arr.ndim}`.")
    return arr


def create_graph_geometry(
    x: TaggedArray,
    arr: jax.Array,
    problem_kind: Literal["linear", "quadratic"],
    problem_shape: Tuple[int, int],
    t: Optional[float],
    epsilon: Union[float, epsilon_scheduler.Epsilon] = None,
    relative_epsilon: Optional[bool] = None,
    scale_cost: Scale_t = 1.0,
    **kwargs: Any,
) -> geometry.Geometry:
    """Create a :class:`ott.geometry.Geometry` from a graph.

    Parameters
    ----------
    x
        Tagged array.
    arr
        Array representation of the graph.
    problem_kind
        Problem kind.
    problem_shape
        Number of source and target points.
    t
        Time parameter of the heat equation.
    epsilon
        Regularization parameter.
    relative_epsilon
        Whether to use relative or absolute epsilon.
    scale_cost
        Scale cost.
    kwargs
        Additional keyword arguments.
    """
    if x.cost == "geodesic":
        if problem_kind == "linear":
            if t is None:
                return geodesic.Geodesic.from_graph(
                    arr, t=epsilon / 4.0, directed=kwargs.pop("directed", True), **kwargs
                )

            n_src, n_tgt = problem_shape
            if n_src + n_tgt != arr.shape[0]:
                raise ValueError(f"Expected `x` to have `{n_src + n_tgt}` points, found `{arr.shape[0]}`.")
            cm = geodesic.Geodesic.from_graph(arr, t=t, directed=kwargs.pop("directed", True), **kwargs).cost_matrix[
                :n_src, n_src:
            ]
            return geometry.Geometry(cm, epsilon=epsilon, relative_epsilon=relative_epsilon, scale_cost=scale_cost)

        if problem_kind == "quadratic":
            n_src, n_tgt = problem_shape
            if n_src + n_tgt != arr.shape[0]:
                raise ValueError(f"Expected `x` to have `{n_src + n_tgt}` points, found `{arr.shape[0]}`.")
            t = epsilon / 4.0 if t is None else t
            cm = geodesic.Geodesic.from_graph(arr, t=t, directed=kwargs.pop("directed", True), **kwargs).cost_matrix[
                :n_src, n_src:
            ]
            return geometry.Geometry(cm, epsilon=epsilon, relative_epsilon=relative_epsilon, scale_cost=scale_cost)

        raise NotImplementedError(f"Invalid problem kind `{problem_kind}`.")
    raise NotImplementedError(f"If the geometry is a graph, `cost` must be `geodesic`, found `{x.cost}`.")
