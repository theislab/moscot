from typing import Any, Dict, Tuple, Callable, Optional, Sequence, TYPE_CHECKING
from functools import partial

from ott.geometry.pointcloud import PointCloud
from ott.tools.sinkhorn_divergence import sinkhorn_divergence
from ott.problems.linear.potentials import DualPotentials
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax.training.train_state import TrainState

from moscot._types import ArrayLike, ScaleCost_t
from moscot._logging import logger

Potential_t = Callable[[jnp.ndarray], float]
CondPotential_t = Callable[[jnp.ndarray, float], float]

if TYPE_CHECKING:
    from ott.geometry import costs


__all__ = ["ConditionalDualPotentials"]


def _compute_sinkhorn_divergence(
    point_cloud_1: ArrayLike,
    point_cloud_2: ArrayLike,
    a: Optional[ArrayLike] = None,
    b: Optional[ArrayLike] = None,
    epsilon: float = 10.0,
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
    input_batch: jnp.ndarray, target: jnp.ndarray, k: int = 30
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Get the k nearest neighbors of the input batch in the target."""
    if target.shape[0] < k:
        raise ValueError(f"k is {k}, but must be smaller or equal than {target.shape[0]}.")
    pairwise_euclidean_distances = jnp.sqrt(jnp.sum((input_batch - target) ** 2, axis=-1))
    negative_distances, indices = jax.lax.top_k(-1 * pairwise_euclidean_distances, k=k)
    return -1 * negative_distances, indices


@jtu.register_pytree_node_class
class ConditionalDualPotentials(DualPotentials):
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

    def transport(self, condition: float, x: jnp.ndarray, forward: bool = True) -> jnp.ndarray:
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
        from ott.geometry import costs
        
        dp = self.to_dual_potentials(condition=condition)
        return dp.transport(x, forward=forward)
        
    def to_dual_potentials(self, condition: ArrayLike) -> DualPotentials:
        """Return the Kantorovich dual potentials from the trained potentials."""

        
        def f(x) -> float:
            return self.state_f.apply_fn({"params": self.state_f.params}, x, condition)

        def g(x) -> float:
            return self.state_g.apply_fn({"params": self.state_g.params}, x, condition)

        return DualPotentials(f, g, corr=True, cost_fn=costs.SqEuclidean())

    
    def distance(self, condition: float, src: ArrayLike, tgt: ArrayLike) -> float:
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
        return [], {"f": self._f, "g": self._g, "cost_fn": self.cost_fn, "corr": self._corr}

    @classmethod
    def tree_unflatten(  # noqa: D102
        cls, aux_data: Dict[str, Any], children: Sequence[Any]
    ) -> "ConditionalDualPotentials":
        return cls(*children, **aux_data)
