from abc import ABC
from enum import Enum
from types import MappingProxyType
from typing import Any, Type, Tuple, Union, Mapping, Optional, NamedTuple

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
        if self.value == Cost.SQEUCL:
            return Euclidean()
        if self.value == Cost.COSINE:
            return Cosine()
        if self.value == Cost.BURES:
            return Bures(**kwargs)
        if self.value == Cost.BUREL_UNBAL:
            return UnbalancedBures(**kwargs)
        raise NotImplementedError(self.value)


class Description(NamedTuple):
    solver: Union[Sinkhorn, LRSinkhorn, GromovWasserstein]
    data: Union[LinearProblem, QuadraticProblem]
    output: Union[Type[SinkhornOutput], Type[LRSinkhornOutput], Type[GWOutput]]


class OTTSolver(BaseSolver, ABC):
    def __init__(self, **kwargs: Any):
        super().__init__()
        self._solver_kwargs = kwargs.copy()

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
        desc: Description,
        output_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> BaseSolverOutput:
        res = desc.solver(desc.data, **kwargs)
        return desc.output(res, **output_kwargs)


class SinkhornSolver(OTTSolver):
    @property
    def problem_kind(self) -> ProblemKind:
        return ProblemKind.LINEAR

    @property
    def _linear_solver(self) -> Union[Sinkhorn, LRSinkhorn]:
        return self._solver

    @_linear_solver.setter
    def _linear_solver(self, solver: Union[Sinkhorn, LRSinkhorn]) -> None:
        self._solver = solver

    # TODO(michalk8): rank
    def _prepare_input(
        self,
        x: Optional[TaggedArray] = None,
        y: Optional[TaggedArray] = None,
        xy: Optional[Tuple[TaggedArray, Optional[TaggedArray]]] = None,
        epsilon: Optional[float] = None,
        online: Union[int, bool] = False,
        scale_cost: Scale_t = None,
        **kwargs: Any,
    ) -> Description:
        geom = self._create_geometry(x, y, epsilon=epsilon, online=online, scale_cost=scale_cost)

        solver = Sinkhorn(**self._solver_kwargs)
        problem = LinearProblem(geom, **kwargs)

        return Description(solver=solver, data=problem, output=SinkhornOutput)


class GWSolver(OTTSolver):
    def _prepare_input(
        self,
        x: Optional[TaggedArray] = None,
        y: Optional[TaggedArray] = None,
        xy: Optional[Tuple[TaggedArray, Optional[TaggedArray]]] = None,
        epsilon: Optional[float] = None,
        online: Union[int, bool] = False,
        scale_cost: Scale_t = None,
        **kwargs: Any,
    ) -> Description:
        geom_x = self._create_geometry(xy[0], epsilon=epsilon, online=online, scale_cost=scale_cost)
        geom_y = self._create_geometry(xy[1], epsilon=epsilon, online=online, scale_cost=scale_cost)

        solver = GromovWasserstein(**self._solver_kwargs)
        problem = QuadraticProblem(geom_x, geom_y, geom_xy=None, fused_penalty=0.0, **kwargs)

        return Description(solver=solver, data=problem, output=GWOutput)

    @property
    def problem_kind(self) -> ProblemKind:
        return ProblemKind.QUAD


class FGWSolver(GWSolver):
    def _prepare_input(
        self,
        x: Optional[TaggedArray] = None,
        y: Optional[TaggedArray] = None,
        xy: Optional[Tuple[TaggedArray, Optional[TaggedArray]]] = None,
        epsilon: Optional[float] = None,
        online: Union[int, bool] = False,
        scale_cost: Scale_t = None,
        alpha: float = 0.5,
        **kwargs: Any,
    ) -> Description:
        description = super()._prepare_input(xy=xy, epsilon=epsilon, online=online, scale_cost=scale_cost)
        # TODO(michalk8): re-add some checks?
        geom_xy = self._create_geometry(x=x, y=y, epsilon=epsilon, online=online, scale_cost=scale_cost)
        self._validate_geoms(description.data.geom_xx, description.data.geom_yy, geom_xy)

        problem = QuadraticProblem(
            description.data.geom_xx,
            description.data.geom_yy,
            geom_xy=geom_xy,
            fused_penalty=self._alpha_to_fused_penalty(alpha),
            **kwargs,
        )

        return Description(solver=description.solver, data=problem, output=description.output)

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
