import abc
import inspect
import types
from typing import Any, Literal, Mapping, Optional, Set, Tuple, Union

import jax
from ott.geometry import costs, epsilon_scheduler, geometry, pointcloud
from ott.problems.linear import linear_problem
from ott.problems.quadratic import quadratic_problem
from ott.solvers.linear import sinkhorn, sinkhorn_lr
from ott.solvers.quadratic import gromov_wasserstein

from moscot._types import ProblemKind_t, QuadInitializer_t, SinkhornInitializer_t
from moscot.backends.ott._utils import alpha_to_fused_penalty, check_shapes, ensure_2d
from moscot.backends.ott.output import OTTOutput
from moscot.base.solver import OTSolver
from moscot.costs import get_cost
from moscot.utils.tagged_array import TaggedArray

__all__ = ["SinkhornSolver", "GWSolver"]

OTTSolver_t = Union[sinkhorn.Sinkhorn, sinkhorn_lr.LRSinkhorn, gromov_wasserstein.GromovWasserstein]
OTTProblem_t = Union[linear_problem.LinearProblem, quadratic_problem.QuadraticProblem]
Scale_t = Union[float, Literal["mean", "median", "max_cost", "max_norm", "max_bound"]]


class OTTJaxSolver(OTSolver[OTTOutput], abc.ABC):
    """Base class for :mod:`ott` solvers :cite:`cuturi2022optimal`.

    Parameters
    ----------
    jit
        Whether to :func:`~jax.jit` the :attr:`solver`.
    """

    def __init__(self, jit: bool = True):
        super().__init__()
        self._solver: Optional[OTTSolver_t] = None
        self._problem: Optional[OTTProblem_t] = None
        self._jit = jit

    def _create_geometry(
        self,
        x: TaggedArray,
        epsilon: Optional[Union[float, epsilon_scheduler.Epsilon]] = None,
        relative_epsilon: Optional[bool] = None,
        scale_cost: Scale_t = 1.0,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> geometry.Geometry:
        if x.is_point_cloud:
            cost_fn = x.cost
            if cost_fn is None:
                cost_fn = costs.SqEuclidean()
            elif isinstance(cost_fn, str):
                cost_fn = get_cost(cost_fn, backend="ott", **kwargs)
            if not isinstance(cost_fn, costs.CostFn):
                raise TypeError(f"Expected `cost_fn` to be `ott.geometry.costs.CostFn`, found `{type(cost_fn)}`.")

            y = None if x.data_tgt is None else ensure_2d(x.data_tgt, reshape=True)
            x = ensure_2d(x.data_src, reshape=True)
            if y is not None and x.shape[1] != y.shape[1]:
                raise ValueError(
                    f"Expected `x/y` to have the same number of dimensions, found `{x.shape[1]}/{y.shape[1]}`."
                )

            return pointcloud.PointCloud(
                x,
                y=y,
                cost_fn=cost_fn,
                epsilon=epsilon,
                relative_epsilon=relative_epsilon,
                scale_cost=scale_cost,
                batch_size=batch_size,
            )

        arr = ensure_2d(x.data_src, reshape=False)
        if x.is_cost_matrix:
            return geometry.Geometry(
                cost_matrix=arr, epsilon=epsilon, relative_epsilon=relative_epsilon, scale_cost=scale_cost
            )
        if x.is_kernel:
            return geometry.Geometry(
                kernel_matrix=arr, epsilon=epsilon, relative_epsilon=relative_epsilon, scale_cost=scale_cost
            )
        raise NotImplementedError(f"Creating geometry from `tag={x.tag!r}` is not yet implemented.")

    def _solve(  # type: ignore[override]
        self,
        prob: OTTProblem_t,
        **kwargs: Any,
    ) -> OTTOutput:
        solver = jax.jit(self.solver) if self._jit else self.solver
        out = solver(prob, **kwargs)
        return OTTOutput(out)

    @property
    def solver(self) -> OTTSolver_t:
        """:mod:`ott` solver."""
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
    """Solver for the :term:`linear problem`.

    The (Kantorovich relaxed) :term:`OT` problem is defined by two distributions in the same space.
    The aim is to obtain a probabilistic map from the source distribution to the target distribution such that
    the (weighted) sum of the distances between coupled data point in the source and the target distribution is
    minimized.

    Parameters
    ----------
    jit
        Whether to :func:`~jax.jit` the :attr:`solver`.
    rank
        Rank of the solver. If `-1`, use :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` :cite:`cuturi:2013`,
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
            kwargs.setdefault("gamma", 10)
            kwargs.setdefault("gamma_rescale", True)
            initializer = "rank2" if initializer is None else initializer
            self._solver = sinkhorn_lr.LRSinkhorn(
                rank=rank, epsilon=epsilon, initializer=initializer, kwargs_init=initializer_kwargs, **kwargs
            )
        else:
            initializer = "default" if initializer is None else initializer
            self._solver = sinkhorn.Sinkhorn(initializer=initializer, kwargs_init=initializer_kwargs, **kwargs)

    def _prepare(
        self,
        xy: Optional[TaggedArray] = None,
        x: Optional[TaggedArray] = None,
        y: Optional[TaggedArray] = None,
        # geometry
        epsilon: Optional[Union[float, epsilon_scheduler.Epsilon]] = None,
        relative_epsilon: Optional[bool] = None,
        batch_size: Optional[int] = None,
        scale_cost: Scale_t = 1.0,
        cost_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        cost_matrix_rank: Optional[int] = None,
        # problem
        **kwargs: Any,
    ) -> linear_problem.LinearProblem:
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
        if cost_matrix_rank is not None:
            geom = geom.to_LRCGeometry(rank=cost_matrix_rank)
        self._problem = linear_problem.LinearProblem(geom, **kwargs)
        return self._problem

    @property
    def xy(self) -> Optional[geometry.Geometry]:
        """Geometry defining the linear term."""
        return None if self._problem is None else self._problem.geom

    @property
    def problem_kind(self) -> ProblemKind_t:  # noqa: D102
        return "linear"

    @classmethod
    def _call_kwargs(cls) -> Tuple[Set[str], Set[str]]:
        geom_kwargs = {"epsilon", "relative_epsilon", "batch_size", "scale_cost", "cost_kwargs", "cost_matrix_rank"}
        problem_kwargs = set(inspect.signature(linear_problem.LinearProblem).parameters.keys())
        problem_kwargs -= {"geom"}
        return geom_kwargs | problem_kwargs, {"epsilon"}


class GWSolver(OTTJaxSolver):
    """Solver for the :term:`quadratic problem` :cite:`memoli:2011`.

    The :term:`Gromov-Wasserstein (GW) <Gromov-Wasserstein>` problem involves two distribution in
    possibly two different spaces. Points in the source distribution are matched to points in the target distribution
    by comparing the relative location of the points within each distribution.

    Parameters
    ----------
    jit
        Whether to :func:`~jax.jit` the :attr:`solver`.
    rank
        Rank of the solver. If `-1` use the full-rank :term:`GW <Gromov-Wasserstein>` :cite:`peyre:2016`,
        otherwise, use the low-rank approach :cite:`scetbon:21b`.
    initializer
        Initializer for :class:`~ott.solvers.quadratic.gromov_wasserstein.GromovWasserstein`.
    initializer_kwargs
        Keyword arguments for the ``initializer``.
    linear_solver_kwargs
        Keyword arguments for :class:`~ott.solvers.linear.sinkhorn.Sinkhorn` or
        :class:`~ott.solvers.linear.sinkhorn_lr.LRSinkhorn`, depending on the ``rank``.
    kwargs
        Keyword arguments for :class:`~ott.solvers.quadratic.gromov_wasserstein.GromovWasserstein` .
    """

    def __init__(
        self,
        jit: bool = True,
        rank: int = -1,
        initializer: QuadInitializer_t = None,
        initializer_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        linear_solver_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        **kwargs: Any,
    ):
        super().__init__(jit=jit)
        if rank > -1:
            linear_solver_kwargs = dict(linear_solver_kwargs)
            linear_solver_kwargs.setdefault("gamma", 10)
            linear_solver_kwargs.setdefault("gamma_rescale", True)
            linear_ot_solver = sinkhorn_lr.LRSinkhorn(rank=rank, **linear_solver_kwargs)
            initializer = "rank2" if initializer is None else initializer
        else:
            linear_ot_solver = sinkhorn.Sinkhorn(**linear_solver_kwargs)
            initializer = None
        self._solver = gromov_wasserstein.GromovWasserstein(
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
        epsilon: Optional[Union[float, epsilon_scheduler.Epsilon]] = None,
        relative_epsilon: Optional[bool] = None,
        batch_size: Optional[int] = None,
        scale_cost: Scale_t = 1.0,
        cost_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        cost_matrix_rank: Optional[int] = None,
        # problem
        alpha: float = 0.5,
        **kwargs: Any,
    ) -> quadratic_problem.QuadraticProblem:
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

        self._problem = quadratic_problem.QuadraticProblem(
            geom_xx, geom_yy, geom_xy, fused_penalty=fused_penalty, **kwargs
        )
        return self._problem

    @property
    def x(self) -> Optional[geometry.Geometry]:
        """The first geometry defining the quadratic term."""
        return None if self._problem is None else self._problem.geom_xx

    @property
    def y(self) -> geometry.Geometry:
        """The second geometry defining the quadratic term."""
        return None if self._problem is None else self._problem.geom_yy

    @property
    def xy(self) -> Optional[geometry.Geometry]:
        """Geometry defining the linear term in the :term:`FGW <fused Gromov-Wasserstein>`."""
        return None if self._problem is None else self._problem.geom_xy

    @property
    def is_fused(self) -> Optional[bool]:
        """Whether the solver is fused."""
        return None if self._problem is None else (self.xy is not None)

    @property
    def problem_kind(self) -> ProblemKind_t:  # noqa: D102
        return "quadratic"

    @classmethod
    def _call_kwargs(cls) -> Tuple[Set[str], Set[str]]:
        geom_kwargs = {"epsilon", "relative_epsilon", "batch_size", "scale_cost", "cost_kwargs", "cost_matrix_rank"}
        problem_kwargs = set(inspect.signature(quadratic_problem.QuadraticProblem).parameters.keys())
        problem_kwargs -= {"geom_xx", "geom_yy", "geom_xy", "fused_penalty"}
        problem_kwargs |= {"alpha"}
        return geom_kwargs | problem_kwargs, {"epsilon"}
