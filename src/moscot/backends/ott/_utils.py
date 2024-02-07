from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import optax
from flax.training.train_state import TrainState

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import scipy.sparse as sp
from ott.geometry import costs, epsilon_scheduler, geometry, pointcloud
from ott.problems.linear.potentials import DualPotentials
from ott.tools.sinkhorn_divergence import sinkhorn_divergence as sinkhorn_div

from moscot._logging import logger
from moscot._types import ArrayLike, ScaleCost_t

Potential_t = Callable[[jnp.ndarray], float]


__all__ = ["ConditionalDualPotentials", "sinkhorn_divergence"]


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
    input_batch: jnp.ndarray, target: jnp.ndarray, k: int = 30
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Get the k nearest neighbors of the input batch in the target."""
    if target.shape[0] < k:
        raise ValueError(f"k is {k}, but must be smaller or equal than {target.shape[0]}.")
    pairwise_euclidean_distances = pointcloud.PointCloud(input_batch, target).cost_matrix
    return jax.lax.approx_min_k(pairwise_euclidean_distances, k=k, recall_target=0.95, aggregate_to_topk=True)


@jtu.register_pytree_node_class
class ConditionalDualPotentials:
    r"""The conditional Kantorovich dual potential functions as introduced in :cite:`bunne2022supervised`.

    :math:`f` and :math:`g` are a pair of functions, candidates for the dual
    OT Kantorovich problem, supposedly optimal for a given pair of measures.

    Parameters
    ----------
    f
        The first conditional dual potential function.
    g
        The second conditional dual potential function.
    cost_fn
        The cost function used to solve the OT problem.
    corr
        Whether the duals solve the problem in distance form, or correlation
        form (as used for instance for ICNNs, see, e.g., top right of p.3 in
        :cite:`makkuva:20`)
    """

    def __init__(self, state_f: TrainState, state_g: TrainState):
        self._state_f = state_f
        self._state_g = state_g

    def transport(self, condition: ArrayLike, x: jnp.ndarray, forward: bool = True) -> jnp.ndarray:
        r"""Conditionally transport ``vec`` according to Brenier formula :cite:`brenier:91`.

        Uses Theorem 1.17 from :cite:`santambrogio:15` to compute an OT map when
        given the Legendre transform of the dual potentials.

        That OT map can be recovered as :math:`x- (\nabla h)^{-1}\circ \nabla f(x)`
        For the case :math:`h(\cdot) = \|\cdot\|^2, \nabla h(\cdot) = 2 \cdot\,`,
        and as a consequence :math:`h^*(\cdot) = \|.\|^2 / 4`, while one has that
        :math:`\nabla h^*(\cdot) = (\nabla h)^{-1}(\cdot) = 0.5 \cdot\,`.

        When the dual potentials are solved in correlation form (only in the Sq.
        Euclidean distance case), the maps are :math:`\nabla g` for forward,
        :math:`\nabla f` for backward.

        Parameters
        ----------
            condition
                Condition for conditional Neural OT.
            vec
                Points to transport, array of shape ``[n, d]``.
            forward
                Whether to transport the points from source to the target
                distribution or vice-versa.

        Returns
        -------
            The transported points.
        """
        dp = self.to_dual_potentials(condition=condition)
        return dp.transport(x, forward=forward)

    def to_dual_potentials(self, condition: ArrayLike) -> DualPotentials:
        """Return the Kantorovich dual potentials from the trained potentials."""

        def f(x, c) -> float:
            return self._state_f.apply_fn({"params": self._state_f.params}, x, c)

        def g(x, c) -> float:
            return self._state_g.apply_fn({"params": self._state_g.params}, x, c)

        return DualPotentials(partial(f, c=condition), partial(g, c=condition), corr=True, cost_fn=costs.SqEuclidean())

    def distance(self, condition: ArrayLike, src: ArrayLike, tgt: ArrayLike) -> float:
        """Evaluate 2-Wasserstein distance between samples using dual potentials.

        Uses Eq. 5 from :cite:`makkuva:20` when given in `corr` form, direct
        estimation by integrating dual function against points when using dual form.

        Parameters
        ----------
        src
            Samples from the source distribution, array of shape ``[n, d]``.
        tgt
            Samples from the target distribution, array of shape ``[m, d]``.

        Returns
        -------
            Wasserstein distance.
        """
        dp = self.to_dual_potentials(condition=condition)
        return dp.distance(src=src, tgt=tgt)

    def get_f(self, condition: ArrayLike) -> DualPotentials:
        """Get the first dual potential function."""
        return lambda x: self._state_f.apply_fn({"params": self._state_f.params}, x=jnp.concatenate(x, condition))

    def get_g(self, condition: ArrayLike) -> Potential_t:
        """Get the second dual potential function."""
        return lambda x: self._state_g.apply_fn({"params": self._state_g.params}, x=jnp.concatenate(x, condition))

    def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:  # noqa: D102
        return [], {"state_f": self._state_f, "state_g": self._state_g}

    @classmethod
    def tree_unflatten(  # noqa: D102
        cls, aux_data: Dict[str, Any], children: Sequence[Any]
    ) -> "ConditionalDualPotentials":
        return cls(*children, **aux_data)

def _get_optimizer(
    learning_rate: float = 1e-4, b1: float = 0.5, b2: float = 0.9, weight_decay: float = 0.0, **kwargs: Any
) -> Type[optax.GradientTransformation]:
    return optax.adamw(learning_rate=learning_rate, b1=b1, b2=b2, weight_decay=weight_decay, **kwargs)


def _compute_metrics_sinkhorn(
    tgt: jnp.ndarray,
    src: jnp.ndarray,
    pred_tgt: jnp.ndarray,
    pred_src: jnp.ndarray,
    valid_eps: float,
    valid_sinkhorn_kwargs: Mapping[str, Any],
) -> Dict[str, float]:
    sinkhorn_loss_data = sinkhorn_div(
        geom=pointcloud.PointCloud,
        x=tgt,
        y=src,
        epsilon=valid_eps,
        sinkhorn_kwargs=valid_sinkhorn_kwargs,
    ).divergence
    sinkhorn_loss_forward = sinkhorn_div(
        geom=pointcloud.PointCloud,
        x=tgt,
        y=pred_tgt,
        epsilon=valid_eps,
        sinkhorn_kwargs=valid_sinkhorn_kwargs,
    ).divergence
    sinkhorn_loss_inverse = sinkhorn_div(
        geom=pointcloud.PointCloud,
        x=src,
        y=pred_src,
        epsilon=valid_eps,
        sinkhorn_kwargs=valid_sinkhorn_kwargs,
    ).divergence
    return {
        "sinkhorn_loss_forward": jnp.abs(sinkhorn_loss_forward),
        "sinkhorn_loss_inverse": jnp.abs(sinkhorn_loss_inverse),
        "sinkhorn_loss_data": jnp.abs(sinkhorn_loss_data),
    }


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
