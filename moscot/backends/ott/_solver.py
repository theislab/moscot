from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, Type, Union, Callable, Optional, NamedTuple
from inspect import signature

from ott.geometry import Grid, Geometry, PointCloud
from ott.core.problems import LinearProblem
from ott.core.sinkhorn import make as make_sinkhorn, Sinkhorn
from ott.geometry.costs import Bures, Cosine, CostFn, Euclidean, UnbalancedBures
from ott.core.sinkhorn_lr import LRSinkhorn
from ott.core.quad_problems import QuadraticProblem
from ott.core.gromov_wasserstein import make as make_gw, GromovWasserstein
import jax.numpy as jnp
import numpy.typing as npt

# TODO(michalk8): initialize ott solvers in init (so that they are not re-jitted
from moscot.solvers._output import BaseSolverOutput
from moscot.backends.ott._output import GWOutput, SinkhornOutput, LRSinkhornOutput
from moscot.solvers._base_solver import BaseSolver
from moscot.solvers._tagged_array import TaggedArray

__all__ = ("Cost", "SinkhornSolver", "GWSolver", "FGWSolver")


class Cost(str, Enum):
    SQEUCL = "sqeucl"
    COSINE = "cosine"
    BURES = "bures"
    BUREL_UNBAL = "bures_unbal"

    def __call__(self, **kwargs: Any) -> CostFn:
        return _losses[self.value](**kwargs)


_losses: Dict[str, Type[CostFn]] = {
    Cost.SQEUCL: Euclidean,
    Cost.COSINE: Cosine,
    Cost.BURES: Bures,
    Cost.BUREL_UNBAL: UnbalancedBures,
}


class SolverDescription(NamedTuple):
    solver: Callable[[Any], Union[Sinkhorn, LRSinkhorn, GromovWasserstein]]
    output: Union[Type[SinkhornOutput], Type[LRSinkhornOutput], Type[GWOutput]]


class GeometryMixin:
    def __init__(self, **kwargs: Any):
        super().__init__()
        self._solver = self._description.solver(**kwargs)

    @property
    @abstractmethod
    def _description(self) -> SolverDescription:
        pass

    def _create_geometry(
        self,
        x: TaggedArray,
        y: Optional[TaggedArray] = None,
        *,
        eps: Optional[float] = None,
        online: bool = False,
    ) -> Geometry:
        # TODO(michalk8): maybe in the future, enable (more) kwargs for PC/Geometry
        if y is not None:
            cost_fn = self._create_cost(x.loss if y.loss is None else y.loss)
            x, y = self._assert2d(x.data), self._assert2d(y.data)
            if x.shape[1] != y.shape[1]:
                raise ValueError("TODO: x/y dimension mismatch")
            return PointCloud(x, y=y, epsilon=eps, cost_fn=cost_fn, online=online)
        if x.is_point_cloud:
            cost_fn = self._create_cost(x.loss)
            return PointCloud(self._assert2d(x.data), epsilon=eps, cost_fn=cost_fn, online=online)
        if x.is_grid:
            cost_fn = self._create_cost(x.loss)
            return Grid(jnp.asarray(x.data), epsilon=eps, cost_fn=cost_fn)
        if x.is_cost_matrix:
            return Geometry(cost_matrix=self._assert2d(x.data, allow_reshape=False), epsilon=eps)
        if x.is_kernel:
            return Geometry(kernel_matrix=self._assert2d(x.loss, allow_reshape=False), epsilon=eps)

        raise NotImplementedError("TODO: invalid tag")

    @staticmethod
    def _assert2d(arr: npt.ArrayLike, *, allow_reshape: bool = True) -> jnp.ndarray:
        arr = jnp.asarray(arr)
        arr = jnp.reshape(arr, (-1, 1)) if (allow_reshape and arr.ndim == 1) else arr
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

    def _solve(
        self,
        data: Union[LinearProblem, QuadraticProblem],
        **kwargs: Any,
    ) -> BaseSolverOutput:
        return self._description.output(self._solver(data, **kwargs))


class RankMixin(GeometryMixin):
    def __init__(self, rank: Optional[int] = None, **kwargs: Any):
        self._rank = rank  # must be se before calling super().__init__
        if rank is not None:
            kwargs["rank"] = rank
        super().__init__(**kwargs)

    @property
    @abstractmethod
    def _linear_solver(self) -> Union[Sinkhorn, LRSinkhorn]:
        pass

    @_linear_solver.setter
    @abstractmethod
    def _linear_solver(self, solver: Union[Sinkhorn, LRSinkhorn]) -> None:
        pass

    @property
    def rank(self) -> Optional[int]:
        return self._rank

    @rank.setter
    def rank(self, value: Optional[int]) -> None:
        if value == self.rank:
            return
        if value is not None:
            assert value > 1, "Rank must be positive."
        self._rank = value

        if value is not None and isinstance(self._linear_solver, LRSinkhorn):
            self._linear_solver.rank = value
            return
        # TODO(michalk8): find a nicer check
        if value is None and type(self._linear_solver) is Sinkhorn:
            return

        [threshold], kwargs = self._linear_solver.tree_flatten()

        clazz = Sinkhorn if value is None else LRSinkhorn
        params = signature(clazz).parameters
        kwargs = {k: v for k, v in kwargs.items() if k in params}
        if value is not None:
            kwargs["rank"] = value

        # TODO(michalk8): warn if using defaults?
        self._linear_solver = clazz(**kwargs, threshold=threshold)

    @property
    def is_low_rank(self) -> bool:
        return self.rank is not None


class SinkhornSolver(RankMixin, BaseSolver):
    @property
    def _linear_solver(self) -> Union[Sinkhorn, LRSinkhorn]:
        return self._solver

    @_linear_solver.setter
    def _linear_solver(self, solver: Union[Sinkhorn, LRSinkhorn]) -> None:
        self._solver = solver

    def _prepare_input(
        self,
        x: TaggedArray,
        y: Optional[TaggedArray] = None,
        eps: Optional[float] = None,
        online: bool = False,
        **kwargs: Any,
    ) -> LinearProblem:
        # case when user switched from FGW -> GW
        kwargs.pop("xx", None)
        kwargs.pop("yy", None)
        if "rank" in kwargs:
            self.rank = kwargs.pop("rank")

        geom = self._create_geometry(x, y, eps=eps, online=online)
        return LinearProblem(geom, **kwargs)

    @property
    def _description(self) -> SolverDescription:
        if self.is_low_rank:
            return SolverDescription(LRSinkhorn, LRSinkhornOutput)
        return SolverDescription(make_sinkhorn, SinkhornOutput)


class GWSolver(RankMixin, BaseSolver):
    def _prepare_input(
        self,
        x: TaggedArray,
        y: Optional[TaggedArray] = None,
        eps: Optional[float] = None,
        online: bool = False,
        **kwargs: Any,
    ) -> QuadraticProblem:
        if y is None:
            raise ValueError("TODO: missing second data")
        # case when user switched from FGW -> GW
        kwargs.pop("xx", None)
        kwargs.pop("yy", None)
<<<<<<< HEAD
        # TODO(michalk8): pass epsilon
        geom_x = self._create_geometry(x, **kwargs)
        geom_y = self._create_geometry(y, **kwargs)

        # TODO(michalk8): marginals + kwargs?
        return QuadraticProblem(geom_x, geom_y, geom_xy=None, fused_penalty=0.0, a=a, b=b,
                                tau_a=tau_a,
                                tau_b=tau_b,
                                )
=======
        if "rank" in kwargs:
            # maybe instantiate the new solver only when the rank is passed
            self.rank = kwargs.pop("rank")

        geom_x = self._create_geometry(x, eps=eps, online=online)
        geom_y = self._create_geometry(y, eps=eps, online=online)
        return QuadraticProblem(geom_x, geom_y, geom_xy=None, fused_penalty=0.0, **kwargs)
>>>>>>> 42055ad21699f9e10f50d4438bd481d4b0899ef7

    @property
    def _linear_solver(self) -> Union[Sinkhorn, LRSinkhorn]:
        return self._solver.linear_ot_solver

    @_linear_solver.setter
    def _linear_solver(self, solver: Union[Sinkhorn, LRSinkhorn]) -> None:
        self._solver.linear_ot_solver = solver

    @property
    def _description(self) -> SolverDescription:
        return SolverDescription(make_gw, GWOutput)


class FGWSolver(GWSolver):
    def _prepare_input(
        self,
        x: TaggedArray,
        y: Optional[TaggedArray] = None,
        xx: Optional[TaggedArray] = None,
        yy: Optional[TaggedArray] = None,
        eps: Optional[float] = None,
        online: bool = False,
        alpha: float = 0.5,
        **kwargs: Any,
    ) -> QuadraticProblem:
        if xx is None:
            raise ValueError("TODO: no array defining joint")
        problem = super()._prepare_input(x, y, eps=eps, online=online)

        if xx.is_cost_matrix or xx.is_kernel:
            # TODO(michalk8): warn if `yy` is not None that we're ignoring it?
            geom_xy = self._create_geometry(xx, eps=eps, online=online)
        elif yy is not None:
            geom_xy = self._create_geometry(xx, yy, eps=eps, online=online)
        else:
            raise ValueError("TODO: specify the 2nd array if this is not kernel/cost")
        self._validate_geoms(problem.geom_xx, problem.geom_yy, geom_xy)

        return QuadraticProblem(
            problem.geom_xx,
            problem.geom_yy,
            geom_xy=geom_xy,
            fused_penalty=self._alpha_to_fused_penalty(alpha),
            **kwargs,
        )

    @staticmethod
    def _validate_geoms(geom_x: Geometry, geom_y: Geometry, geom_xy: Geometry) -> None:
        # TODO(michalk8): check if this is right
        if geom_x.shape[0] != geom_xy.shape[0]:
            raise ValueError("TODO: first and joint geom mismatch")
        if geom_y.shape[0] != geom_xy.shape[1]:
            raise ValueError("TODO: second and joint geom mismatch")

    @staticmethod
    def _alpha_to_fused_penalty(alpha: float) -> float:
        assert 0 < alpha < 1, "TODO: alpha must be in (0, 1)"
        return (1 - alpha) / alpha
