from abc import abstractmethod
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Type, Tuple, Union, Mapping, Optional, NamedTuple
from inspect import signature

from typing_extensions import Literal

from ott.geometry import Grid, Geometry, PointCloud
from ott.core.problems import LinearProblem
from ott.core.sinkhorn import Sinkhorn
from ott.geometry.costs import Bures, Cosine, CostFn, Euclidean, UnbalancedBures
from ott.core.sinkhorn_lr import LRSinkhorn
from ott.core.quad_problems import QuadraticProblem
from ott.core.gromov_wasserstein import GromovWasserstein
import jax.numpy as jnp
import numpy.typing as npt

from moscot.solvers._output import BaseSolverOutput
from moscot.backends.ott._output import GWOutput, SinkhornOutput, LRSinkhornOutput
from moscot.solvers._base_solver import BaseSolver, ProblemKind
from moscot.solvers._tagged_array import TaggedArray

__all__ = ("Cost", "SinkhornSolver", "GWSolver", "FGWSolver")

Scale_t = Optional[Union[float, Literal["mean", "median", "max_cost", "max_norm", "max_bound"]]]


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
    solver: Union[Type[Sinkhorn], Type[LRSinkhorn], Type[GromovWasserstein]]
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
        epsilon: Optional[float] = None,
        online: Union[int, bool] = False,
        scale_cost: Scale_t = None,
    ) -> Geometry:
        # TODO(michalk8): maybe in the future, enable (more) kwargs for PC/Geometry
        if y is not None:
            cost_fn = self._create_cost(x.loss if y.loss is None else y.loss)
            x, y = self._assert2d(x.data), self._assert2d(y.data)
            if x.shape[1] != y.shape[1]:
                raise ValueError("TODO: x/y dimension mismatch")
            return PointCloud(x, y=y, epsilon=epsilon, cost_fn=cost_fn, online=online, scale_cost=scale_cost)
        if x.is_point_cloud:
            cost_fn = self._create_cost(x.loss)
            return PointCloud(
                self._assert2d(x.data), epsilon=epsilon, cost_fn=cost_fn, online=online, scale_cost=scale_cost
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
                kernel_matrix=self._assert2d(x.loss, allow_reshape=False), epsilon=epsilon, scale_cost=scale_cost
            )

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
        output_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> BaseSolverOutput:
        return self._description.output(self._solver(data, **kwargs), **output_kwargs)


class RankMixin(GeometryMixin):
    def __init__(self, rank: int = -1, **kwargs: Any):
        # a bit ugly - must be set before calling super().__init__
        # otherwise, would need to refactor how description works
        self._rank = max(-1, rank)
        if rank > 0:
            kwargs["rank"] = rank
        super().__init__(**kwargs)
        self._other_solver: Union[LRSinkhorn, Sinkhorn] = self._create_other_solver()

    @property
    @abstractmethod
    def _linear_solver(self) -> Union[Sinkhorn, LRSinkhorn]:
        pass

    @_linear_solver.setter
    @abstractmethod
    def _linear_solver(self, solver: Union[Sinkhorn, LRSinkhorn]) -> None:
        pass

    def _solve(
        self,
        data: Union[LinearProblem, QuadraticProblem],
        output_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> BaseSolverOutput:
        return super()._solve(data, output_kwargs={**output_kwargs, "rank": self.rank}, **kwargs)

    def _create_other_solver(self) -> Union[Sinkhorn, LRSinkhorn]:
        new = Sinkhorn if isinstance(self._linear_solver, LRSinkhorn) else LRSinkhorn
        [threshold], kwargs = self._linear_solver.tree_flatten()
        actual_params = signature(new).parameters
        kwargs = {k: v for k, v in kwargs.items() if k in actual_params}
        if new is LRSinkhorn:
            kwargs["rank"] = 42  # dummy value, updated when setting rank
            kwargs["implicit_diff"] = False  # implicit diff. not yet implemented for LRSink

        return new(threshold=threshold, **kwargs)

    @property
    def rank(self) -> int:
        return self._rank

    @rank.setter
    def rank(self, rank: Optional[int]) -> None:
        if rank is None:  # TODO(michalk8): remove me once writing middle-end tests
            rank = -1
        rank = max(-1, rank)
        if rank == self.rank:
            return

        if self.is_low_rank:
            if rank > 0:  # update the rank
                self._linear_solver.rank = rank
            else:  # or just swap the solvers
                self._linear_solver, self._other_solver = self._other_solver, self._linear_solver
            self._rank = rank
            return

        # we're not LR, but the other solver is
        self._linear_solver, self._other_solver = self._other_solver, self._linear_solver
        self._linear_solver.rank = rank
        self._rank = rank

    def _set_ctx(self, _: Any, **kwargs: Any) -> Any:
        if "rank" not in kwargs:
            return self.rank
        old_rank, self.rank = self.rank, kwargs.pop("rank")
        return old_rank

    def _reset_ctx(self, rank: Optional[int]) -> None:
        self.rank = rank

    @property
    def is_low_rank(self) -> bool:
        return self.rank > 0


class SinkhornSolver(RankMixin, BaseSolver):
    @property
    def problem_kind(self) -> ProblemKind:
        return ProblemKind.LINEAR

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
        epsilon: Optional[float] = None,
        online: Union[int, bool] = False,
        scale_cost: Scale_t = None,
        **kwargs: Any,
    ) -> LinearProblem:
        kwargs.pop("rank", None)  # set in context afterwards
        geom = self._create_geometry(x, y, epsilon=epsilon, online=online, scale_cost=scale_cost)
        return LinearProblem(geom, **kwargs)

    @property
    def _description(self) -> SolverDescription:
        if self.is_low_rank:
            return SolverDescription(LRSinkhorn, LRSinkhornOutput)
        return SolverDescription(Sinkhorn, SinkhornOutput)


class GWSolver(RankMixin, BaseSolver):
    def _prepare_input(
        self,
        x: TaggedArray,
        y: Optional[TaggedArray] = None,
        epsilon: Optional[float] = None,
        online: Union[int, bool] = False,
        scale_cost: Scale_t = None,
        **kwargs: Any,
    ) -> QuadraticProblem:
        kwargs.pop("rank", None)  # set in context afterwards
        geom_x = self._create_geometry(x, epsilon=epsilon, online=online, scale_cost=scale_cost)
        geom_y = self._create_geometry(y, epsilon=epsilon, online=online, scale_cost=scale_cost)
        return QuadraticProblem(geom_x, geom_y, geom_xy=None, fused_penalty=0.0, **kwargs)

    @property
    def problem_kind(self) -> ProblemKind:
        return ProblemKind.QUAD

    @RankMixin.rank.setter
    def rank(self, rank: int) -> None:
        RankMixin.rank.fset(self, rank)  # correctly sets rank for `_linear_solver`
        # TODO(michalk8): make a PR to OTT that simplifies how rank is stored
        self._solver.rank = rank  # sets rank of the GW solver (see above TODO)

    @property
    def epsilon(self) -> float:
        return self._solver.epsilon

    @epsilon.setter
    def epsilon(self, epsilon: Optional[float]):
        if epsilon is None:
            epsilon = 1e-2
        self._solver.epsilon = epsilon
        if self.is_low_rank:
            self._linear_solver.epsilon = epsilon

    @property
    def _linear_solver(self) -> Union[Sinkhorn, LRSinkhorn]:
        return self._solver.linear_ot_solver

    @_linear_solver.setter
    def _linear_solver(self, solver: Union[Sinkhorn, LRSinkhorn]) -> None:
        self._solver.linear_ot_solver = solver

    def _set_ctx(self, data: Any, **kwargs: Any) -> Tuple[Optional[int], Optional[int]]:
        if "epsilon" not in kwargs:
            old_epsilon = self.epsilon
            old_ctx = super()._set_ctx(data, **kwargs)
        else:
            # important: to first set rank, then epsilon; the former changes the solver
            old_ctx = super()._set_ctx(data, **kwargs)
            old_epsilon, self.epsilon = self.epsilon, kwargs.pop("epsilon")
        return old_ctx, old_epsilon

    def _reset_ctx(self, rank_epsilon: Tuple[Optional[int], Optional[float]]) -> None:
        rank, epsilon = rank_epsilon
        # important: to first set rank, then epsilon; the former changes the solver
        super()._reset_ctx(rank)
        self.epsilon = epsilon

    @property
    def _description(self) -> SolverDescription:
        return SolverDescription(GromovWasserstein, GWOutput)


class FGWSolver(GWSolver):
    def _prepare_input(
        self,
        x: TaggedArray,
        y: Optional[TaggedArray] = None,
        xx: Optional[TaggedArray] = None,
        yy: Optional[TaggedArray] = None,
        epsilon: Optional[float] = None,
        online: Union[int, bool] = False,
        scale_cost: Scale_t = None,
        alpha: float = 0.5,
        rank: int = None,
        **kwargs: Any,
    ) -> QuadraticProblem:
        problem = super()._prepare_input(x, y, epsilon=epsilon, online=online, scale_cost=scale_cost)
        if xx.is_cost_matrix or xx.is_kernel:
            # TODO(michalk8): warn if `yy` is not None that we're ignoring it?
            geom_xy = self._create_geometry(xx, epsilon=epsilon, online=online, scale_cost=scale_cost)
        elif yy is not None:
            geom_xy = self._create_geometry(xx, yy, epsilon=epsilon, online=online, scale_cost=scale_cost)
        else:
            raise ValueError("TODO: specify the 2nd array if this is not kernel/cost")
        self._validate_geoms(problem.geom_xx, problem.geom_yy, geom_xy)

        kwargs.pop("rank", None)  # set in context afterwards
        return QuadraticProblem(
            problem.geom_xx,
            problem.geom_yy,
            geom_xy=geom_xy,
            fused_penalty=self._alpha_to_fused_penalty(alpha),
            **kwargs,
        )

    @property
    def problem_kind(self) -> ProblemKind:
        return ProblemKind.QUAD_FUSED

    @staticmethod
    def _validate_geoms(geom_x: Geometry, geom_y: Geometry, geom_xy: Geometry) -> None:
        if geom_x.shape[0] != geom_xy.shape[0]:
            raise ValueError("TODO: first and joint geom mismatch")
        if geom_y.shape[0] != geom_xy.shape[1]:
            raise ValueError("TODO: second and joint geom mismatch")

    @staticmethod
    def _alpha_to_fused_penalty(alpha: float) -> float:
        assert 0 < alpha < 1, "TODO: alpha must be in (0, 1)"
        return (1 - alpha) / alpha
