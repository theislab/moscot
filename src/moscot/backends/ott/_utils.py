from collections import defaultdict
from functools import partial
from typing import Any, Dict, Iterable, Literal, Optional, Tuple, Union

import jax
import jax.experimental.sparse as jesp
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from ott.geometry import epsilon_scheduler, geodesic, geometry, pointcloud
from ott.initializers.linear import initializers as init_lib
from ott.initializers.linear import initializers_lr as lr_init_lib
from ott.neural import datasets
from ott.solvers import utils as solver_utils
from ott.tools.sinkhorn_divergence import sinkhorn_divergence as sinkhorn_div

from moscot._logging import logger
from moscot._types import ArrayLike, ScaleCost_t

Scale_t = Union[float, Literal["mean", "median", "max_cost", "max_norm", "max_bound"]]


__all__ = ["sinkhorn_divergence"]


class InitializerResolver:
    """Class for creating various OT solver initializers.

    This class provides static methods to create and manage different types of
    initializers used in optimal transport solvers, including low-rank, k-means,
    and standard Sinkhorn initializers.
    """

    @staticmethod
    def lr_from_str(
        initializer: str,
        rank: int,
        **kwargs: Any,
    ) -> lr_init_lib.LRInitializer:
        """Create a low-rank initializer from a string specification.

        Parameters
        ----------
        initializer : str
            Either existing initializer instance or string specifier.
        rank : int
            Rank for the initialization.
        **kwargs : Any
            Additional keyword arguments for initializer creation.

        Returns
        -------
        LRInitializer
            Configured low-rank initializer.

        Raises
        ------
        NotImplementedError
            If requested initializer type is not implemented.
        """
        if isinstance(initializer, lr_init_lib.LRInitializer):
            return initializer
        if initializer == "k-means":
            return lr_init_lib.KMeansInitializer(rank=rank, **kwargs)
        if initializer == "generalized-k-means":
            return lr_init_lib.GeneralizedKMeansInitializer(rank=rank, **kwargs)
        if initializer == "random":
            return lr_init_lib.RandomInitializer(rank=rank, **kwargs)
        if initializer == "rank2":
            return lr_init_lib.Rank2Initializer(rank=rank, **kwargs)
        raise NotImplementedError(f"Initializer `{initializer}` is not implemented.")

    @staticmethod
    def from_str(
        initializer: str,
        **kwargs: Any,
    ) -> init_lib.SinkhornInitializer:
        """Create a Sinkhorn initializer from a string specification.

        Parameters
        ----------
        initializer : str
            String specifier for initializer type.
        **kwargs : Any
            Additional keyword arguments for initializer creation.

        Returns
        -------
        SinkhornInitializer
            Configured Sinkhorn initializer.

        Raises
        ------
        NotImplementedError
            If requested initializer type is not implemented.
        """
        if isinstance(initializer, init_lib.SinkhornInitializer):
            return initializer
        if initializer == "default":
            return init_lib.DefaultInitializer(**kwargs)
        if initializer == "gaussian":
            return init_lib.GaussianInitializer(**kwargs)
        if initializer == "sorting":
            return init_lib.SortingInitializer(**kwargs)
        if initializer == "subsample":
            return init_lib.SubsampleInitializer(**kwargs)
        raise NotImplementedError(f"Initializer `{initializer}` is not yet implemented.")


def sinkhorn_divergence(
    point_cloud_1: ArrayLike,
    point_cloud_2: ArrayLike,
    a: Optional[ArrayLike] = None,
    b: Optional[ArrayLike] = None,
    epsilon: Union[float, epsilon_scheduler.Epsilon] = 1e-1,
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

    output = sinkhorn_div(
        pointcloud.PointCloud,
        x=point_cloud_1,
        y=point_cloud_2,
        batch_size=batch_size,
        a=a,
        b=b,
        scale_cost=scale_cost,
        epsilon=epsilon,
        solve_kwargs={
            "tau_a": tau_a,
            "tau_b": tau_b,
        },
        **kwargs,
    )[1]
    xy_conv, xx_conv, *yy_conv = output.converged

    if not xy_conv:
        logger.warning("Solver did not converge in the `x/y` term.")
    if not xx_conv:
        logger.warning("Solver did not converge in the `x/x` term.")
    if len(yy_conv) and not yy_conv[0]:
        logger.warning("Solver did not converge in the `y/y` term.")

    return float(output.divergence)


@partial(jax.jit, static_argnames=["k"])
def get_nearest_neighbors(
    input_batch: jnp.ndarray,
    target: jnp.ndarray,
    k: int = 30,
    recall_target: float = 0.95,
    aggregate_to_topk: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Get the k nearest neighbors of the input batch in the target."""
    if target.shape[0] < k:
        raise ValueError(f"k is {k}, but must be smaller or equal than {target.shape[0]}.")
    pairwise_euclidean_distances = pointcloud.PointCloud(input_batch, target).cost_matrix
    return jax.lax.approx_min_k(
        pairwise_euclidean_distances, k=k, recall_target=recall_target, aggregate_to_topk=aggregate_to_topk
    )


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


def densify(arr: ArrayLike) -> jax.Array:
    """If the input is sparse, convert it to dense.

    Parameters
    ----------
    arr
        Array to check.

    Returns
    -------
    dense :mod:`jax` array.
    """
    if sp.issparse(arr):
        arr = arr.toarray()  # type: ignore[attr-defined]
    elif isinstance(arr, jesp.BCOO):
        arr = arr.todense()
    return jnp.asarray(arr)


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
    if reshape and arr.ndim == 1:
        return jnp.reshape(arr, (-1, 1))
    if arr.ndim != 2:
        raise ValueError(f"Expected array to have 2 dimensions, found `{arr.ndim}`.")
    return arr.astype(jnp.float64)


def convert_scipy_sparse(arr: Union[sp.spmatrix, jesp.BCOO]) -> jesp.BCOO:
    """If the input is a scipy sparse matrix, convert it to a jax BCOO."""
    if sp.issparse(arr):
        return jesp.BCOO.from_scipy_sparse(arr)
    return arr


def _instantiate_geodesic_cost(
    arr: jax.Array,
    problem_shape: Tuple[int, int],
    t: Optional[float],
    is_linear_term: bool,
    epsilon: Union[float, epsilon_scheduler.Epsilon] = None,
    relative_epsilon: Optional[bool] = None,
    scale_cost: Scale_t = 1.0,
    directed: bool = True,
    **kwargs: Any,
) -> geometry.Geometry:
    n_src, n_tgt = problem_shape
    if is_linear_term and n_src + n_tgt != arr.shape[0]:
        raise ValueError(f"Expected `x` to have `{n_src + n_tgt}` points, found `{arr.shape[0]}`.")
    t = epsilon / 4.0 if t is None else t
    cm_full = geodesic.Geodesic.from_graph(arr, t=t, directed=directed, **kwargs).cost_matrix
    cm = cm_full[:n_src, n_src:] if is_linear_term else cm_full
    return geometry.Geometry(cm, epsilon=epsilon, relative_epsilon=relative_epsilon, scale_cost=scale_cost)


def data_match_fn(
    src_lin: Optional[jnp.ndarray] = None,
    tgt_lin: Optional[jnp.ndarray] = None,
    src_quad: Optional[jnp.ndarray] = None,
    tgt_quad: Optional[jnp.ndarray] = None,
    *,
    typ: Literal["lin", "quad", "fused"],
    **data_match_fn_kwargs,
) -> jnp.ndarray:
    if typ == "lin":
        return solver_utils.match_linear(x=src_lin, y=tgt_lin, **data_match_fn_kwargs)
    if typ == "quad":
        return solver_utils.match_quadratic(xx=src_quad, yy=tgt_quad, **data_match_fn_kwargs)
    if typ == "fused":
        return solver_utils.match_quadratic(xx=src_quad, yy=tgt_quad, x=src_lin, y=tgt_lin, **data_match_fn_kwargs)
    raise NotImplementedError(f"Unknown type: {typ}.")


class Loader:

    def __init__(self, dataset: datasets.OTDataset, batch_size: int, seed: Optional[int] = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self._rng = np.random.default_rng(seed)

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, jnp.ndarray]:
        data = defaultdict(list)
        for _ in range(self.batch_size):
            ix = self._rng.integers(0, len(self.dataset))
            for k, v in self.dataset[ix].items():
                data[k].append(v)
        return {k: jnp.vstack(v) for k, v in data.items()}

    def __len__(self):
        return len(self.dataset)


class MultiLoader:
    """Dataset for OT problems with conditions.

    This data loader wraps several data loaders and samples from them.

    Args:
      datasets: Datasets to sample from.
      seed: Random seed.
    """

    def __init__(
        self,
        datasets: Iterable[Loader],
        seed: Optional[int] = None,
    ):
        self.datasets = tuple(datasets)
        self._rng = np.random.default_rng(seed)
        self._iterators: list[MultiLoader] = []
        self._it = 0

    def __next__(self) -> Dict[str, jnp.ndarray]:
        self._it += 1

        ix = self._rng.choice(len(self._iterators))
        iterator = self._iterators[ix]
        if self._it < len(self):
            return next(iterator)
        # reset the consumed iterator and return it's first element
        self._iterators[ix] = iterator = iter(self.datasets[ix])
        return next(iterator)

    def __iter__(self) -> "MultiLoader":
        self._it = 0
        self._iterators = [iter(ds) for ds in self.datasets]
        return self

    def __len__(self) -> int:
        return max((len(ds) for ds in self.datasets), default=0)
