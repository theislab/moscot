import inspect
from functools import partial
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
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
from ott.geometry import costs, epsilon_scheduler, geometry
from ott.geometry.pointcloud import PointCloud
from ott.problems.linear import linear_problem
from ott.problems.linear.potentials import DualPotentials
from ott.solvers.linear import sinkhorn
from ott.tools.sinkhorn_divergence import sinkhorn_divergence

from moscot._logging import logger
from moscot._types import ArrayLike, ScaleCost_t
from moscot.backends.ott._icnn import ICNN

Potential_t = Callable[[jnp.ndarray], float]
CondPotential_t = Callable[[jnp.ndarray, float], float]

if TYPE_CHECKING:
    from ott.geometry import costs


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
        # Note that here the order of f and g should be correct already, as it's swapped in the initialization of CondDualPotentials

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

    @property
    def f(self, condition: ArrayLike) -> DualPotentials:
        """The first dual potential function."""
        return lambda x: self._state_f.apply_fn({"params": self._state_f.params}, x=jnp.concatenate(x, condition))

    @property
    def g(self, condition: ArrayLike) -> CondPotential_t:
        """The second dual potential function."""
        return lambda x: self._state_g.apply_fn({"params": self._state_g.params}, x=jnp.concatenate(x, condition))

    def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:  # noqa: D102
        return [], {"state_f": self._state_f, "state_g": self._state_g}

    @classmethod
    def tree_unflatten(  # noqa: D102
        cls, aux_data: Dict[str, Any], children: Sequence[Any]
    ) -> "ConditionalDualPotentials":
        return cls(*children, **aux_data)


def _get_icnn(
    input_dim: int,
    cond_dim: int,
    pos_weights: bool = False,
    dim_hidden: Iterable[int] = (64, 64, 64, 64),
    **kwargs: Any,
) -> ICNN:
    return ICNN(input_dim=input_dim, cond_dim=cond_dim, pos_weights=pos_weights, dim_hidden=dim_hidden, **kwargs)


def _get_optimizer(
    learning_rate: float = 1e-3, b1: float = 0.5, b2: float = 0.9, weight_decay: float = 0.0, **kwargs: Any
) -> Type[optax.GradientTransformation]:
    return optax.adamw(learning_rate=learning_rate, b1=b1, b2=b2, weight_decay=weight_decay, **kwargs)


# Compute the difference in drug signatures
@jax.jit
def compute_ds_diff(control, treated, push_fwd):
    """Compute Drug Signature difference as the norm between the vector of means of features."""
    base = control.mean(0)

    true = treated.mean(0) - base
    pred = push_fwd.mean(0) - base

    return jnp.linalg.norm(true - pred)


def mmd_rbf(x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Compute MMD between x and y via RBF kernel."""
    x_norm = jnp.square(x).sum(-1)
    xx = jnp.einsum("ia, ja- > ij", x, x)
    x_sq_dist = x_norm[..., :, None] + x_norm[..., None, :] - 2 * xx
    y_norm = jnp.square(y).sum(-1)
    yy = jnp.einsum("ia, ja -> ij", y, y)
    y_sq_dist = y_norm[..., :, None] + y_norm[..., None, :] - 2 * yy
    zz = jnp.einsum("ia, ja -> ij", x, y)
    z_sq_dist = x_norm[..., :, None] + y_norm[..., None, :] - 2 * zz
    var = jnp.var(z_sq_dist)
    XX, YY, XY = (jnp.zeros(xx.shape), jnp.zeros(yy.shape), jnp.zeros(zz.shape))
    array_sum = jnp.sum(y_sq_dist)
    jnp.isnan(array_sum)
    bandwidth_range = [0.5, 0.1, 0.01, 0.005]
    for scale in bandwidth_range:
        XX += jnp.exp(-0.5 * x_sq_dist / (var * scale))
        YY += jnp.exp(-0.5 * y_sq_dist / (var * scale))
        XY += jnp.exp(-0.5 * z_sq_dist / (var * scale))
    return jnp.mean(XX) + jnp.mean(YY) - 2.0 * jnp.mean(XY)


def _regularized_wasserstein(
    x: jnp.ndarray,
    y: jnp.ndarray,
    sinkhorn_kwargs: Mapping[str, Any] = MappingProxyType({}),
    geometry_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> Optional[float]:
    """
    Compute a regularized Wasserstein distance to be used as the fitting term in the loss.
    Fitting term computes how far the predicted target is from teh actual target (ground truth).
    """
    geom = PointCloud(x=x, y=y, **geometry_kwargs)
    return sinkhorn.Sinkhorn(**sinkhorn_kwargs)(linear_problem.LinearProblem(geom)).reg_ot_cost


def _compute_metrics_sinkhorn(
    tgt: jnp.ndarray,
    src: jnp.ndarray,
    pred_tgt: jnp.ndarray,
    pred_src: jnp.ndarray,
    valid_eps: float,
    valid_sinkhorn_kwargs: Mapping[str, Any],
) -> Dict[str, float]:
    sinkhorn_loss_data = sinkhorn_divergence(
        geom=PointCloud,
        x=tgt,
        y=src,
        epsilon=valid_eps,
        sinkhorn_kwargs=valid_sinkhorn_kwargs,
    ).divergence
    sinkhorn_loss_forward = sinkhorn_divergence(
        geom=PointCloud,
        x=tgt,
        y=pred_tgt,
        epsilon=valid_eps,
        sinkhorn_kwargs=valid_sinkhorn_kwargs,
    ).divergence
    sinkhorn_loss_inverse = sinkhorn_divergence(
        geom=PointCloud,
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
