import abc
import inspect
import types
from typing import Any, Literal, Mapping, Optional, Set, Tuple, Union

import jax
import jax.numpy as jnp
import scipy.sparse as sp
from ott.geometry import costs
from ott.geometry.epsilon_scheduler import Epsilon
from ott.geometry.geometry import Geometry
from ott.geometry.pointcloud import PointCloud
from ott.problems.linear.linear_problem import LinearProblem
from ott.problems.quadratic.quadratic_problem import QuadraticProblem
from ott.solvers.linear.sinkhorn import Sinkhorn
from ott.solvers.linear.sinkhorn_lr import LRSinkhorn
from ott.solvers.quadratic.gromov_wasserstein import GromovWasserstein

from moscot._types import (
    ArrayLike,
    ProblemKind_t,
    QuadInitializer_t,
    SinkhornInitializer_t,
)
from moscot.backends.ott._utils import alpha_to_fused_penalty, check_shapes
from moscot.backends.ott.output import OTTOutput
from moscot.base.solver import OTSolver
from moscot.costs import get_cost
from moscot.utils.tagged_array import TaggedArray

__all__ = ["SinkhornSolver", "GWSolver"]

Scale_t = Union[float, Literal["mean", "median", "max_cost", "max_norm", "max_bound"]]


class OTTJaxSolver(OTSolver[OTTOutput], abc.ABC):
    """Base class for :mod:`ott` solvers :cite:`cuturi2022optimal`.

    Parameters
    ----------
    jit
        Whether to jit the :attr:`solver`.
    """

    def __init__(self, jit: bool = True):
        super().__init__()
        self._solver: Optional[Union[Sinkhorn, LRSinkhorn, GromovWasserstein]] = None
        self._problem: Optional[Union[LinearProblem, QuadraticProblem]] = None
        self._jit = jit

    def _create_geometry(
        self,
        x: TaggedArray,
        epsilon: Optional[Union[float, Epsilon]] = None,
        relative_epsilon: Optional[bool] = None,
        scale_cost: Scale_t = 1.0,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> Geometry:
        if x.is_point_cloud:
            cost_fn = x.cost
            if cost_fn is None:
                cost_fn = costs.SqEuclidean()
            elif isinstance(cost_fn, str):
                cost_fn = get_cost(cost_fn, backend="ott", **kwargs)
            if not isinstance(cost_fn, costs.CostFn):
                raise TypeError("TODO")

            y = None if x.data_tgt is None else self._ensure_2d(x.data_tgt, reshape=True)
            x = self._ensure_2d(x.data_src, reshape=True)
            if y is not None and x.shape[1] != y.shape[1]:
                raise ValueError(
                    f"Expected `x/y` to have the same number of dimensions, found `{x.shape[1]}/{y.shape[1]}`."
                )

            return PointCloud(
                x,
                y=y,
                cost_fn=cost_fn,
                epsilon=epsilon,
                relative_epsilon=relative_epsilon,
                scale_cost=scale_cost,
                batch_size=batch_size,
            )

        arr = self._ensure_2d(x.data_src, reshape=False)
        if x.is_cost_matrix:
            return Geometry(cost_matrix=arr, epsilon=epsilon, relative_epsilon=relative_epsilon, scale_cost=scale_cost)
        if x.is_kernel:
            return Geometry(
                kernel_matrix=arr, epsilon=epsilon, relative_epsilon=relative_epsilon, scale_cost=scale_cost
            )
        raise NotImplementedError(f"Creating geometry from `tag={x.tag!r}` is not yet implemented.")

    def _solve(  # type: ignore[override]
        self,
        prob: Union[LinearProblem, QuadraticProblem],
        **kwargs: Any,
    ) -> OTTOutput:
        solver = jax.jit(self.solver) if self._jit else self.solver
        out = solver(prob, **kwargs)
        return OTTOutput(out)

    @staticmethod
    def _ensure_2d(arr: ArrayLike, *, reshape: bool = False) -> Optional[jnp.ndarray]:  # type: ignore[name-defined]
        if sp.issparse(arr):
            arr = arr.A  # type: ignore[attr-defined]
        arr = jnp.asarray(arr)
        if reshape and arr.ndim == 1:
            return jnp.reshape(arr, (-1, 1))
        if arr.ndim != 2:
            raise ValueError(f"Expected array to have 2 dimensions, found `{arr.ndim}`.")
        return arr

    @property
    def solver(self) -> Union[Sinkhorn, LRSinkhorn, GromovWasserstein]:
        """Underlying :mod:`ott` solver."""
        return self._solver

    @property
    def rank(self) -> int:
        """Rank of the :attr:`solver`."""
        return getattr(self.solver, "rank", -1)

    @property
    def is_low_rank(self) -> bool:
        """Whether the :attr:`solver` is low-rank."""
        return self.rank > -1


class SinkhornSolver(OTTJaxSolver):
    """Linear optimal transport problem solver.

    The (Kantorovich relaxed) optimal transport problem is defined by two distributions in the same space.
    The aim is to obtain a probabilistic map from the source distribution to the target distribution such that
    the (weighted) sum of the distances between coupled data point in the source and the target distribution is
    minimized.

    Parameters
    ----------
    rank
        Rank of the linear solver. If `-1`, use :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` :cite:`cuturi:2013`,
        otherwise, use :class:`~ott.solvers.linear.sinkhorn_lr.LRSinkhorn` :cite:`scetbon:21a`.
    epsilon
        Additional epsilon regularization for the low-rank approach.
    initializer
        Initializer for :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` or
        :class:`~ott.solvers.linear.sinkhorn_lr.LRSinkhorn`, depending on the ``rank``.
    initializer_kwargs
        Keyword arguments for the initializer.
    kwargs
        Keyword arguments for :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` or
        :class:`~ott.solvers.linear.sinkhorn_lr.LRSinkhorn`, depending on the ``rank``.
    """

    def __init__(
        self,
        jit: bool = True,
        rank: int = -1,
        epsilon: float = 0.0,
        initializer: SinkhornInitializer_t = None,
        initializer_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        **kwargs: Any,
    ):
        super().__init__(jit=jit)
        if rank > -1:
            initializer = "rank2" if initializer is None else initializer
            self._solver = LRSinkhorn(
                rank=rank, epsilon=epsilon, initializer=initializer, kwargs_init=initializer_kwargs, **kwargs
            )
        else:
            initializer = "default" if initializer is None else initializer
            self._solver = Sinkhorn(initializer=initializer, kwargs_init=initializer_kwargs, **kwargs)

    def _prepare(
        self,
        xy: Optional[TaggedArray] = None,
        x: Optional[TaggedArray] = None,
        y: Optional[TaggedArray] = None,
        # geometry
        epsilon: Optional[Union[float, Epsilon]] = None,
        relative_epsilon: Optional[bool] = None,
        batch_size: Optional[int] = None,
        scale_cost: Scale_t = 1.0,
        cost_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        cost_matrix_rank: Optional[int] = None,
        # problem
        **kwargs: Any,
    ) -> LinearProblem:
        del x, y
        if xy is None:
            raise ValueError(f"Unable to create geometry from `xy={xy}`.")

        geom = self._create_geometry(
            xy,
            epsilon=epsilon,
            relative_epsilon=relative_epsilon,
            batch_size=batch_size,
            scale_cost=scale_cost,
            **cost_kwargs,
        )
        # TODO(MUCDK): do we really want to do this? solver rank/geom rank are 2 different things
        if self.is_low_rank:
            rank = self.rank if cost_matrix_rank is None else cost_matrix_rank
            # batch_size cannot be passed in this function
            geom = geom.to_LRCGeometry(rank=rank)
        self._problem = LinearProblem(geom, **kwargs)
        return self._problem

    @property
    def xy(self) -> Optional[Geometry]:
        """Geometry defining the linear term."""
        return None if self._problem is None else self._problem.geom

    @property
    def problem_kind(self) -> ProblemKind_t:  # noqa: D102
        return "linear"

    @classmethod
    def _call_kwargs(cls) -> Tuple[Set[str], Set[str]]:
        geom_kwargs = {"epsilon", "relative_epsilon", "batch_size", "scale_cost", "cost_kwargs", "cost_matrix_rank"}
        problem_kwargs = set(inspect.signature(LinearProblem).parameters.keys())
        problem_kwargs -= {"geom"}
        return geom_kwargs | problem_kwargs, {"epsilon"}


class GWSolver(OTTJaxSolver):
    """Solver solving quadratic optimal transport problem.

    The Gromov-Wasserstein (GW) problem involves two distribution in possibly two different spaces.
    Points in the source distribution are matched to points in the target distribution by comparing the relative
    location of the points within each distribution.

    Parameters
    ----------
    rank
        Rank of the quadratic solver. If `-1` use the full-rank GW :cite:`memoli:2011`,
        otherwise, use the low-rank approach :cite:`scetbon:21b`.
    gamma
        Only in low-rank setting: the (inverse of the) gradient step size used by the mirror descent algorithm
        :cite:`scetbon:22b`.
    gamma_rescale
        Only in low-rank setting: whether to rescale `gamma` every iteration as described in :cite:`scetbon:22b`.
    initializer
        Initializer for :class:`~ott.solvers.quadratic.gromov_wasserstein.GromovWasserstein`.
    initializer_kwargs
        Keyword arguments for the initializer.
    linear_solver_kwargs
        Keyword arguments for :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` or
        :class:`~ott.solvers.linear.sinkhorn_lr.LRSinkhorn`, depending on the ``rank``.
    kwargs
        Keyword arguments for :class:`~ott.solvers.quadratic.gromov_wasserstein.GromovWasserstein` .
    """

    def __init__(
        self,
        rank: int = -1,
        gamma: float = 10.0,
        gamma_rescale: bool = True,
        initializer: QuadInitializer_t = None,
        initializer_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        linear_solver_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        **kwargs: Any,
    ):
        super().__init__()
        if rank > -1:
            linear_ot_solver = LRSinkhorn(rank=rank, gamma=gamma, gamma_rescale=gamma_rescale, **linear_solver_kwargs)
            initializer = "rank2" if initializer is None else initializer
        else:
            linear_ot_solver = Sinkhorn(**linear_solver_kwargs)
            initializer = None
        self._solver = GromovWasserstein(
            rank=rank,
            linear_ot_solver=linear_ot_solver,
            quad_initializer=initializer,
            kwargs_init=initializer_kwargs,
            **kwargs,
        )

    def _prepare(
        self,
        xy: Optional[TaggedArray] = None,
        x: Optional[TaggedArray] = None,
        y: Optional[TaggedArray] = None,
        # geometry
        epsilon: Optional[Union[float, Epsilon]] = None,
        relative_epsilon: Optional[bool] = None,
        batch_size: Optional[int] = None,
        scale_cost: Scale_t = 1.0,
        cost_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        cost_matrix_rank: Optional[int] = None,
        # problem
        alpha: float = 0.5,
        **kwargs: Any,
    ) -> QuadraticProblem:
        if x is None or y is None:
            raise ValueError(f"Unable to create geometry from `x={x}`, `y={y}`.")
        geom_kwargs: Any = {
            "epsilon": epsilon,
            "relative_epsilon": relative_epsilon,
            "batch_size": batch_size,
            "scale_cost": scale_cost,
            "cost_matrix_rank": cost_matrix_rank,
            **cost_kwargs,
        }
        geom_xx = self._create_geometry(x, **geom_kwargs)
        geom_yy = self._create_geometry(y, **geom_kwargs)
        if alpha == 1.0 or xy is None:  # GW
            # arbitrary fused penalty; must be positive
            geom_xy, fused_penalty = None, 1.0
        else:  # FGW
            fused_penalty = alpha_to_fused_penalty(alpha)
            geom_xy = self._create_geometry(xy, **geom_kwargs)
            check_shapes(geom_xx, geom_yy, geom_xy)

        self._problem = QuadraticProblem(geom_xx, geom_yy, geom_xy, fused_penalty=fused_penalty, **kwargs)
        return self._problem

    @property
    def x(self) -> Optional[Geometry]:
        """First geometry defining the quadratic term."""
        return None if self._problem is None else self._problem.geom_xx

    @property
    def y(self) -> Geometry:
        """Second geometry defining the quadratic term."""
        return None if self._problem is None else self._problem.geom_yy

    @property
    def xy(self) -> Optional[Geometry]:
        """Geometry defining the linear term in the fused case."""
        return None if self._problem is None else self._problem.geom_xy

    @property
    def is_fused(self) -> Optional[bool]:
        """Whether the problem is fused."""
        return None if self._problem is None else (self.xy is not None)

    @property
    def problem_kind(self) -> ProblemKind_t:  # noqa: D102
        return "quadratic"

    @classmethod
    def _call_kwargs(cls) -> Tuple[Set[str], Set[str]]:
        geom_kwargs = {"epsilon", "relative_epsilon", "batch_size", "scale_cost", "cost_kwargs", "cost_matrix_rank"}
        problem_kwargs = set(inspect.signature(QuadraticProblem).parameters.keys())
        problem_kwargs -= {"geom_xx", "geom_yy", "geom_xy", "fused_penalty"}
        problem_kwargs |= {"alpha"}
        return geom_kwargs | problem_kwargs, {"epsilon"}
