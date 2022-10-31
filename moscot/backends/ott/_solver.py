from abc import ABC
from types import MappingProxyType
from typing import Any, Union, Literal, Mapping, Optional

from scipy.sparse import issparse

from ott.geometry import Epsilon, Geometry, PointCloud
from ott.core.sinkhorn import Sinkhorn
from ott.geometry.costs import Bures, Cosine, CostFn, Euclidean, SqEuclidean, UnbalancedBures
from ott.core.was_solver import WassersteinSolver
from ott.core.sinkhorn_lr import LRSinkhorn
from ott.core.quad_problems import QuadraticProblem
from ott.core.linear_problems import LinearProblem
from ott.core.gromov_wasserstein import GromovWasserstein
import jax.numpy as jnp

from moscot._types import ArrayLike
from moscot._constants._enum import ModeEnum
from moscot.backends.ott._output import OTTOutput
from moscot.problems.base._utils import _filter_kwargs
from moscot.solvers._base_solver import OTSolver, ProblemKind
from moscot.solvers._tagged_array import TaggedArray

__all__ = ["OTTCost", "SinkhornSolver", "GWSolver", "FGWSolver"]

Scale_t = Union[float, Literal["mean", "median", "max_cost", "max_norm", "max_bound"]]
Epsilon_t = Union[float, Epsilon]


class OTTCost(ModeEnum):
    EUCL = "euclidean"
    SQEUCL = "sq_euclidean"
    COSINE = "cosine"
    BURES = "bures"
    BUREL_UNBAL = "unbalanced_bures"

    def __call__(self, **kwargs: Any) -> CostFn:
        if self.value == OTTCost.EUCL:
            return Euclidean()
        if self.value == OTTCost.SQEUCL:
            return SqEuclidean()
        if self.value == OTTCost.COSINE:
            return Cosine()
        if self.value == OTTCost.BURES:
            return Bures(**kwargs)
        if self.value == OTTCost.BUREL_UNBAL:
            return UnbalancedBures(**kwargs)
        raise NotImplementedError(self.value)


class OTTJaxSolver(OTSolver[OTTOutput], ABC):  # noqa: B024
    """Base class for :mod:`ott` solvers :cite:`cuturi2022optimal`."""

    def __init__(self):
        super().__init__()
        self._solver: Optional[Union[Sinkhorn, LRSinkhorn, GromovWasserstein]] = None
        self._problem: Optional[Union[LinearProblem, QuadraticProblem]] = None

    def _create_geometry(
        self,
        x: TaggedArray,
        **kwargs: Any,
    ) -> Geometry:
        if x.is_point_cloud:
            kwargs = _filter_kwargs(PointCloud, Geometry, **kwargs)
            cost_fn = self._create_cost(x.cost)
            x, y = self._assert2d(x.data_src), self._assert2d(x.data_tgt)
            n, m = x.shape[1], (None if y is None else y.shape[1])
            if m is not None and n != m:
                raise ValueError(f"Expected `x/y` to have the same number of dimensions, found `{n}/{m}`.")
            return PointCloud(x, y=y, cost_fn=cost_fn, **kwargs)  # TODO: add ScaleCost

        kwargs = _filter_kwargs(Geometry, **kwargs)
        arr = self._assert2d(x.data_src, allow_reshape=False)
        if x.is_cost_matrix:
            return Geometry(cost_matrix=arr, **kwargs)
        if x.is_cost_matrix:
            return Geometry(kernel_matrix=arr, **kwargs)
        raise NotImplementedError(f"Creating geometry from `tag={x.tag!r}` is not yet implemented.")

    def _solve(  # type: ignore[override]
        self,
        prob: Union[LinearProblem, QuadraticProblem],
        **kwargs: Any,
    ) -> OTTOutput:
        out = self.solver(prob, **kwargs)
        return OTTOutput(out)

    @staticmethod
    def _assert2d(arr: Optional[ArrayLike], *, allow_reshape: bool = True) -> Optional[jnp.ndarray]:
        if arr is None:
            return None
        arr: jnp.ndarray = jnp.asarray(arr.A if issparse(arr) else arr)  # type: ignore[attr-defined, no-redef]
        if allow_reshape and arr.ndim == 1:
            return jnp.reshape(arr, (-1, 1))
        if arr.ndim != 2:
            raise ValueError(f"Expected array to have 2 dimensions, found `{arr.ndim}`.")
        return arr

    @staticmethod
    def _create_cost(cost: Optional[Union[str, CostFn]], **kwargs: Any) -> CostFn:
        if isinstance(cost, CostFn):
            return cost
        if cost is None:
            cost = "sq_euclidean"
        return OTTCost(cost)(**kwargs)

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
        Rank of the linear solver. If `-1`, use :class:`~ott.core.sinkhorn.Sinkhorn` :cite:`cuturi:2013`,
        otherwise, use :class:`~ott.core.sinkhorn_lr.LRSinkhorn` :cite:`scetbon:21a`.
    initializer_kwargs
        Keyword arguments for the initializer.
    kwargs
        Keyword arguments for :class:`~ott.core.sinkhorn.Sinkhorn` or :class:`~ott.core.sinkhorn_lr.LRSinkhorn`,
        depending on the ``rank``.
    """

    def __init__(self, rank: int = -1, initializer_kwargs: Mapping[str, Any] = MappingProxyType({}), **kwargs: Any):
        super().__init__()
        if rank > -1:
            kwargs = _filter_kwargs(Sinkhorn, LRSinkhorn, **kwargs)
            self._solver = LRSinkhorn(rank=rank, kwargs_init=initializer_kwargs, **kwargs)
        else:
            kwargs = _filter_kwargs(Sinkhorn, **kwargs)
            self._solver = Sinkhorn(kwargs_init=initializer_kwargs, **kwargs)

    def _prepare(
        self,
        xy: Optional[TaggedArray] = None,
        x: Optional[TaggedArray] = None,
        y: Optional[TaggedArray] = None,
        epsilon: Optional[Epsilon_t] = None,
        batch_size: Optional[int] = None,
        scale_cost: Scale_t = 1.0,
        **kwargs: Any,
    ) -> LinearProblem:
        del x, y
        if xy is None:
            raise ValueError(f"Unable to create geometry from `xy={xy}`.")

        geom = self._create_geometry(xy, epsilon=epsilon, batch_size=batch_size, scale_cost=scale_cost, **kwargs)
        kwargs = _filter_kwargs(LinearProblem, **kwargs)
        self._problem = LinearProblem(geom, **kwargs)

        return self._problem

    @property
    def xy(self) -> Optional[Geometry]:
        """Geometry defining the linear term."""
        return None if self._problem is None else self._problem.geom

    @property
    def problem_kind(self) -> ProblemKind:
        return ProblemKind.LINEAR


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
    initializer_kwargs
        Keyword arguments for the initializer.
    linear_solver_kwargs
        Keyword arguments for :class:`~ott.core.sinkhorn.Sinkhorn` or :class:`~ott.core.sinkhorn_lr.LRSinkhorn`,
        depending on the ``rank``.
    kwargs
        Keyword arguments for :class:`~ott.core.gromov_wasserstein.GromovWasserstein` .
    """

    def __init__(
        self,
        rank: int = -1,
        initializer_kwargs: Mapping[str, Any] = MappingProxyType({}),
        linear_solver_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ):
        super().__init__()
        if "initializer" in kwargs:  # rename arguments
            kwargs["quad_initializer"] = kwargs.pop("initializer")
        if rank > -1:
            linear_ot_solver = LRSinkhorn(
                rank=rank, **linear_solver_kwargs
            )  # initialization handled by quad_initializer
        else:
            linear_ot_solver = Sinkhorn(**linear_solver_kwargs)  # initialization handled by quad_initializer
        kwargs = _filter_kwargs(GromovWasserstein, WassersteinSolver, **kwargs)
        self._solver = GromovWasserstein(
            rank=rank, linear_ot_solver=linear_ot_solver, kwargs_init=initializer_kwargs, **kwargs
        )

    def _prepare(
        self,
        xy: Optional[TaggedArray] = None,
        x: Optional[TaggedArray] = None,
        y: Optional[TaggedArray] = None,
        **kwargs: Any,
    ) -> QuadraticProblem:
        del xy
        if x is None or y is None:
            raise ValueError(f"Unable to create geometry from `x={x}`, `y={y}`.")

        geom_x = self._create_geometry(x, **kwargs)
        geom_y = self._create_geometry(y, **kwargs)

        kwargs = _filter_kwargs(QuadraticProblem, **kwargs)
        self._problem = QuadraticProblem(geom_x, geom_y, geom_xy=None, **kwargs)
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
    def problem_kind(self) -> ProblemKind:
        return ProblemKind.QUAD


class FGWSolver(GWSolver):
    """
    Class which solves quadratic OT problem with a linear term included.

    The Fused Gromov-Wasserstein (FGW) problem involves two distributions living in two subspaces,
    corresponding to the linear term and the quadratic term, respectively.

    The subspace corresponding to the linear term is shared between the two distributions.
    The subspace corresponding to the quadratic term is defined in possibly two different spaces.
    The matching obtained from the FGW is a compromise between the ones induced by the linear OT problem and
    the quadratic OT problem :cite:`vayer:2018`.

    This solver wraps :class:`~ott.core.gromov_wasserstein.GromovWasserstein` with a non-trivial ``fused_penalty``.

    Parameters
    ----------
    rank
        Rank of the quadratic solver. If `-1` use the full-rank GW :cite:`memoli:2011`,
        otherwise, use the low-rank approach :cite:`scetbon:21b`.
    initializer_kwargs
        Keyword arguments for the initializer.
    linear_solver_kwargs
        Keyword arguments for :class:`~ott.core.sinkhorn.Sinkhorn` or :class:`~ott.core.sinkhorn_lr.LRSinkhorn`,
        depending on the ``rank``.
    kwargs
        Keyword arguments for :class:`~ott.core.gromov_wasserstein.GromovWasserstein` .
    """

    def _prepare(
        self,
        xy: Optional[TaggedArray] = None,
        x: Optional[TaggedArray] = None,
        y: Optional[TaggedArray] = None,
        alpha: float = 0.5,
        **kwargs: Any,
    ) -> QuadraticProblem:
        if xy is None:
            raise ValueError(f"Unable to create geometry from `xy={xy}`.")

        prob = super()._prepare(x=x, y=y, **kwargs)
        geom_xy = self._create_geometry(xy, **kwargs)
        self._validate_geoms(prob.geom_xx, prob.geom_yy, geom_xy)

        kwargs = _filter_kwargs(QuadraticProblem, **kwargs)
        self._problem = QuadraticProblem(
            geom_xx=prob.geom_xx,
            geom_yy=prob.geom_yy,
            geom_xy=geom_xy,
            fused_penalty=self._alpha_to_fused_penalty(alpha),
            **kwargs,
        )
        return self._problem

    @property
    def xy(self) -> Optional[Geometry]:
        """Geometry defining the linear term."""
        return None if self._problem is None else self._problem.geom_xy

    @property
    def problem_kind(self) -> ProblemKind:
        return ProblemKind.QUAD_FUSED

    @staticmethod
    def _validate_geoms(geom_x: Geometry, geom_y: Geometry, geom_xy: Geometry) -> None:
        n, m = geom_xy.shape
        n_, m_ = geom_x.shape[0], geom_y.shape[0]
        if n != n_:
            raise ValueError(f"Expected the first geometry to have `{n}` points, found `{n_}`.")
        if m != m_:
            raise ValueError(f"Expected the second geometry to have `{m}` points, found `{m_}`.")

    @staticmethod
    def _alpha_to_fused_penalty(alpha: float) -> float:
        if not (0 < alpha <= 1):
            raise ValueError(f"Expected `alpha` to be in interval `(0, 1]`, found `{alpha}`.")
        return (1 - alpha) / alpha
