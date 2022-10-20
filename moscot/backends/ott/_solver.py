from abc import ABC
from enum import Enum
from typing import Any, Union, Literal, Optional

from ott.core import initializers as init_lib
from ott.geometry import Grid, Epsilon, Geometry, PointCloud
from ott.core.sinkhorn import Sinkhorn
from ott.geometry.costs import Bures, Cosine, CostFn, SqEuclidean, UnbalancedBures
from ott.core.sinkhorn_lr import LRSinkhorn
from ott.core.quad_problems import QuadraticProblem
from ott.core.linear_problems import LinearProblem
from ott.core.gromov_wasserstein import GromovWasserstein
import jax.numpy as jnp

from moscot._types import ArrayLike
from moscot._docs._docs import d
from moscot.backends.ott._output import OTTOutput
from moscot.solvers._base_solver import OTSolver, ProblemKind
from moscot.solvers._tagged_array import TaggedArray

__all__ = ["Cost", "SinkhornSolver", "GWSolver", "FGWSolver"]

Scale_t = Union[float, Literal["mean", "median", "max_cost", "max_norm", "max_bound"]]
Epsilon_t = Union[float, Epsilon]


class Cost(str, Enum):
    SQEUCL = "sqeucl"
    COSINE = "cosine"
    BURES = "bures"
    BUREL_UNBAL = "bures_unbal"

    def __call__(self, **kwargs: Any) -> CostFn:
        if self.value == Cost.SQEUCL:
            return SqEuclidean()
        if self.value == Cost.COSINE:
            return Cosine()
        if self.value == Cost.BURES:
            return Bures(**kwargs)
        if self.value == Cost.BUREL_UNBAL:
            return UnbalancedBures(**kwargs)
        raise NotImplementedError(self.value)


# TODO(michalk8): consider removing the variadic parametrization in the future
class OTTJaxSolver(OTSolver[OTTOutput], ABC):  # noqa: B024
    """
    Class handling the preparation of :class:`ott.geometry.Geometry`.

    Parameters
    ----------
    kwargs
        keyword arguments for one of the following:

            - :class:`ott.core.sinkhorn.Sinkhorn`
            - :class:`ott.core.sinkhorn_lr.LRSinkhorn`
            - :class:`ott.core.gromov_wasserstein.GromovWasserstein`
    """

    def __init__(self) -> None:
        super().__init__()
        self._solver: Optional[Union[Sinkhorn, LRSinkhorn, GromovWasserstein]] = None
        self._problem: Optional[Union[LinearProblem, QuadraticProblem]] = None

    def _create_geometry(
        self,
        x: TaggedArray,
        *,
        epsilon: Optional[Epsilon_t] = None,
        batch_size: Optional[int] = None,
        scale_cost: Scale_t = 1.0,
    ) -> Geometry:
        """
        TODO.

        Raises
        ------
        ValueError
            If the dimensions of `x` and `y` do not match
        NotImplementedError
            If the `tag` is not among the implemented ones
        """
        # TODO(michalk8): maybe in the future, enable (more) kwargs for PC/Geometry
        if x.is_point_cloud:
            cost_fn = self._create_cost(x.loss)
            x, y = self._assert2d(x.data), self._assert2d(x.data_y)
            if y is not None and x.shape[1] != y.shape[1]:  # type: ignore[attr-defined]
                raise ValueError("TODO: x/y dimension mismatch")
            return PointCloud(x, y=y, epsilon=epsilon, cost_fn=cost_fn, batch_size=batch_size, scale_cost=scale_cost)
        if x.is_point_cloud:
            cost_fn = self._create_cost(x.loss)
            return PointCloud(
                self._assert2d(x.data), epsilon=epsilon, cost_fn=cost_fn, batch_size=batch_size, scale_cost=scale_cost
            )
        if x.is_grid:
            cost_fn = self._create_cost(x.loss)
            return Grid(jnp.asarray(x.data), epsilon=epsilon, cost_fn=cost_fn, scale_cost=scale_cost)
        if x.is_cost_matrix:
            return Geometry(
                cost_matrix=self._assert2d(x.data, allow_reshape=False), epsilon=epsilon, scale_cost=scale_cost
            )
        if x.is_kernel:
            return Geometry(
                kernel_matrix=self._assert2d(x.data, allow_reshape=False), epsilon=epsilon, scale_cost=scale_cost
            )

        raise NotImplementedError("TODO: invalid tag")

    def _solve(  # type: ignore[override]
        self,
        prob: Union[LinearProblem, QuadraticProblem],
        **kwargs: Any,
    ) -> OTTOutput:
        out = self.solver(prob, **kwargs)
        return OTTOutput(out)

    @staticmethod
    def _assert2d(arr: Optional[ArrayLike], *, allow_reshape: bool = True) -> jnp.ndarray:
        if arr is None:
            return None
        arr: jnp.ndarray = jnp.asarray(arr)  # type: ignore[no-redef]
        if allow_reshape and arr.ndim == 1:
            return jnp.reshape(arr, (-1, 1))
        if arr.ndim != 2:
            raise ValueError("TODO: expected 2D")
        return arr

    @staticmethod
    def _create_cost(cost: Optional[Union[str, CostFn]], **kwargs: Any) -> CostFn:
        if isinstance(cost, CostFn):
            return cost
        if cost is None:
            cost = "sqeucl"
        return Cost(cost)(**kwargs)

    @property
    def solver(self) -> Union[Sinkhorn, LRSinkhorn, GromovWasserstein]:
        """Underlying optimal transport solver."""
        return self._solver

    @property
    def rank(self) -> int:
        """Rank of the :attr:`solver`."""
        return getattr(self.solver, "rank", -1)

    @property
    def is_low_rank(self) -> bool:
        """Whether the :attr:`solver` is low-rank."""
        return self.rank > -1


@d.dedent
class SinkhornSolver(OTTJaxSolver):
    """
    Solver class solving linear Optimal Transport problems.

    The (Kantorovich relaxed) Optimal Transport problem is defined by two distributions in the same space. The
    aim is to obtain a probabilistic map from the source distribution to the target distribution such that
    the (weighted) sum of the distances between coupled data point in the source and the target distribution is
    minimized.

    This solver wraps :class:`ott.core.sinkhorn.Sinkhorn` :cite:`cuturi:2013` by default and :cite:`cuturi:2013`
    :class:`ott.core.sinkhorn_lr.LRSinkhorn` :cite:`scetbon:21` if `rank` is a positive integer. In the
    former case, the solver makes use of the Sinkhorn algorithm, in the latter a mirror descent algorithm.
    TODO: link notebooks for example

    Parameters
    ----------
    %(OTSolver.parameters)s
    """

    def __init__(self, rank: int = -1, **kwargs: Any):
        super().__init__()
        initializer = kwargs.pop("initializer", None)
        if rank > -1:
            if initializer is None:
                initializer = "random"
            self._solver = LRSinkhorn(rank=rank, initializer=initializer, **kwargs)
        else:
            if initializer is None:
                initializer = init_lib.DefaultInitializer()
            self._solver = Sinkhorn(initializer=initializer, **kwargs)

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
        if xy is None:
            raise ValueError("TODO")

        geom = self._create_geometry(xy, epsilon=epsilon, batch_size=batch_size, scale_cost=scale_cost)
        self._problem = LinearProblem(geom, **kwargs)

        return self._problem

    @property
    def xy(self) -> Optional[Geometry]:
        return None if self._problem is None else self._problem.geom

    @property
    def problem_kind(self) -> ProblemKind:
        return ProblemKind.LINEAR


@d.dedent
class GWSolver(OTTJaxSolver):
    """
    Solver class solving quadratic Optimal Transport problems.

    The Gromov-Wasserstein (GW) problem involves two distribution in possibly two different spaces. Points in the source
    distribution are matched to points in the target distribution by comparing the relative location of the datapoints
    within each distribution.

    This solver wraps :class:`ott.core.gromov_wasserstein.GromovWasserstein` which handles both the full rank
    Gromov-Wasserstein algorithm :cite:`memoli:2011` as well as the low rank approach :cite:`scetbon:21b`.
    In both cases the solver makes use of a mirror-descent algorithm :cite:`memoli:2011`.

    TODO: link notebooks for example

    Parameters
    ----------
    %(OTSolver.parameters)s
    """

    def __init__(self, **kwargs: Any):
        super().__init__()
        quad_initializer = kwargs.pop("initializer", None)  # OTT-JAX allows for "None" as initializer
        kwargs_init = kwargs.pop("initializer", None)
        self._solver = GromovWasserstein(quad_initializer=quad_initializer, kwargs_init=kwargs_init, **kwargs)

    def _prepare(
        self,
        xy: Optional[TaggedArray] = None,
        x: Optional[TaggedArray] = None,
        y: Optional[TaggedArray] = None,
        epsilon: Optional[Epsilon_t] = None,
        batch_size: Optional[int] = None,
        scale_cost: Scale_t = 1.0,
        **kwargs: Any,
    ) -> QuadraticProblem:
        if x is None or y is None:
            raise ValueError("TODO")

        geom_x = self._create_geometry(x, epsilon=epsilon, batch_size=batch_size, scale_cost=scale_cost)
        geom_y = self._create_geometry(y, epsilon=epsilon, batch_size=batch_size, scale_cost=scale_cost)

        if epsilon is not None:
            self.solver.epsilon = epsilon
        self._problem = QuadraticProblem(geom_x, geom_y, geom_xy=None, **kwargs)

        return self._problem

    @property
    def x(self) -> Optional[Geometry]:
        """Geometry of the first space."""
        return None if self._problem is None else self._problem.geom_xx

    @property
    def y(self) -> Geometry:
        """Geometry of the second space."""
        return None if self._problem is None else self._problem.geom_yy

    @property
    def problem_kind(self) -> ProblemKind:
        return ProblemKind.QUAD


class FGWSolver(GWSolver):
    """
    Class which solves quadratic OT problems with a linear term included.

    The Fused Gromov-Wasserstein (FGW) problem involves two distributions living in two subspaces,
    corresponding to the linear term and the quadratic termm, respectively. The subspace corresponding
    to the linear term is shared between the two distributions. The subspace corresponding to the quadratic
    term is defined in possibly two different spaces. The matchings obtained from FGW are a compromise
    between the ones induced by the linear OT problem and the purely quadratic OT problem (GW) :cite:`vayer:2018`.

    This solver wraps :class:`ott.core.gromov_wasserstein.GromovWasserstein` with non-trivial `fused_penalty`.

    TODO: link notebooks for example

    Parameters
    ----------
    %(GWSolver.parameters)s
    """

    def _prepare(
        self,
        xy: Optional[TaggedArray] = None,
        x: Optional[TaggedArray] = None,
        y: Optional[TaggedArray] = None,
        epsilon: Optional[Epsilon_t] = None,
        batch_size: Optional[int] = None,
        scale_cost: Scale_t = 1.0,
        alpha: float = 0.5,
        **kwargs: Any,
    ) -> QuadraticProblem:
        if xy is None:
            raise ValueError("TODO")

        prob = super()._prepare(x=x, y=y, epsilon=epsilon, batch_size=batch_size, scale_cost=scale_cost)
        geom_xy = self._create_geometry(xy, epsilon=epsilon, batch_size=batch_size, scale_cost=scale_cost)
        self._validate_geoms(prob.geom_xx, prob.geom_yy, geom_xy)

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
        """Geometry of the joint space."""
        return None if self._problem is None else self._problem.geom_xy

    @property
    def problem_kind(self) -> ProblemKind:
        return ProblemKind.QUAD_FUSED

    @staticmethod
    def _validate_geoms(geom_x: Geometry, geom_y: Geometry, geom_xy: Geometry) -> None:
        if geom_x.shape[0] != geom_xy.shape[0]:
            raise ValueError(f"TODO: first and joint geom mismatch: `{geom_x.shape}`, `{geom_xy.shape}`")
        if geom_y.shape[0] != geom_xy.shape[1]:
            raise ValueError(f"TODO: second and joint geom mismatch: `{geom_y.shape}`, `{geom_xy.shape}`")

    @staticmethod
    def _alpha_to_fused_penalty(alpha: float) -> float:
        assert 0 < alpha < 1, "TODO: alpha must be in (0, 1)"
        return (1 - alpha) / alpha
