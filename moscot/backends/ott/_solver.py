from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, Type, Tuple, Union, Optional

from ott.geometry import Grid, Geometry, PointCloud
from ott.core.problems import LinearProblem
from ott.core.sinkhorn import make as Sinkhorn
from ott.geometry.costs import Bures, Cosine, CostFn, Euclidean, UnbalancedBures
from ott.core.sinkhorn_lr import LRSinkhorn as SinkhornLR
from ott.core.quad_problems import QuadraticProblem
from ott.core.gromov_wasserstein import make as GW
import jax.numpy as jnp
import numpy.typing as npt

from moscot.solvers._data import TaggedArray

# TODO(michalk8): initialize ott solvers in init (so that they are not re-jitted
from moscot.solvers._output import BaseSolverOutput
from moscot.backends.ott._output import GWOutput, SinkhornOutput, LRSinkhornOutput
from moscot.solvers._base_solver import BaseSolver


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


class GeometryMixin:
    def __init__(self, **kwargs: Any):
        super().__init__()
        self._solver = self._solver_output[0](**kwargs)

    @property
    @abstractmethod
    def _solver_output(self) -> Tuple[Type[Any], Type[BaseSolverOutput]]:
        pass

    def _create_geometry(
        self,
        x: TaggedArray,
        y: Optional[TaggedArray] = None,
        *,
        eps: Optional[float] = None,
        **kwargs: Any,
    ) -> Geometry:
        if y is not None:
            # TODO: pass cost kwargs
            cost_fn = self._create_cost(x.loss if y.loss is None else y.loss)
            x, y = self._assert2d(x.data), self._assert2d(y.data)
            if x.shape[1] != y.shape[1]:
                raise ValueError("TODO: x/y dimension mismatch")
            return PointCloud(x, y=y, epsilon=eps, cost_fn=cost_fn, **kwargs)
        if x.is_point_cloud:
            # TODO: pass cost kwargs
            cost_fn = self._create_cost(x.loss)
            return PointCloud(self._assert2d(x.data), epsilon=eps, cost_fn=cost_fn, **kwargs)
        if x.is_grid:
            # TODO: pass cost kwargs
            cost_fn = self._create_cost(x.loss)
            return Grid(jnp.asarray(x.data), epsilon=eps, cost_fn=cost_fn, **kwargs)
        if x.is_cost_matrix:
            return Geometry(cost_matrix=self._assert2d(x.data, allow_reshape=False), epsilon=eps, **kwargs)
        if x.is_kernel:
            return Geometry(kernel_matrix=self._assert2d(x.loss, allow_reshape=False), epsilon=eps, **kwargs)

        raise NotImplementedError("TODO: invalid tag")

    @staticmethod
    def _set_eps(
        problem: Union[LinearProblem, QuadraticProblem], eps: Optional[float] = None
    ) -> Union[LinearProblem, QuadraticProblem]:
        if eps is None:
            # TODO(michalk8): mb. the below code also works for this case
            return problem

        eps_geom = Geometry(epsilon=eps)
        if isinstance(problem, LinearProblem):
            problem.geom.copy_epsilon(eps_geom)
            return problem
        if isinstance(problem, QuadraticProblem):
            problem.geom_xx.copy_epsilon(eps_geom)
            problem.geom_yy.copy_epsilon(eps_geom)
            if problem.is_fused:
                problem.geom_xy.copy_epsilon(eps_geom)
            return problem
        raise TypeError("TODO: expected OTT problem")

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

    def _solve(self, data: Union[LinearProblem, QuadraticProblem], **kwargs: Any) -> BaseSolverOutput:
        return self._solver_output[1](self._solver(data, **kwargs))


class SinkhornSolver(GeometryMixin, BaseSolver):
    def _prepare_input(
        self,
        x: TaggedArray,
        y: Optional[TaggedArray] = None,
        a: Optional[npt.ArrayLike] = None,
        b: Optional[npt.ArrayLike] = None,
        tau_a: Optional[float] = 1.0,
        tau_b: Optional[float] = 1.0,
        **kwargs: Any,
    ) -> LinearProblem:
        kwargs.pop("xx", None)
        kwargs.pop("yy", None)
        geom = self._create_geometry(x, y, **kwargs)
        return LinearProblem(geom, a=a, b=b, tau_a=tau_a, tau_b=tau_b)

    @property
    def _solver_output(self) -> Type[BaseSolverOutput]:
        return Sinkhorn, SinkhornOutput


class LRSinkhornSolver(SinkhornSolver):
    def _solve(self, data: Union[LinearProblem, QuadraticProblem], **kwargs: Any) -> BaseSolverOutput:
        return self._solver_output[1](self._solver(data, **kwargs), threshold=self._solver.threshold)

    @property
    def _solver_output(self) -> Tuple[Type[Any], Type[BaseSolverOutput]]:
        return SinkhornLR, LRSinkhornOutput


class GWSolver(GeometryMixin, BaseSolver):
    def _prepare_input(
        self,
        x: TaggedArray,
        y: Optional[TaggedArray] = None,
        a: Optional[npt.ArrayLike] = None,
        b: Optional[npt.ArrayLike] = None,
        tau_a: Optional[float] = 1.0,
        tau_b: Optional[float] = 1.0,
        **kwargs: Any,
    ) -> QuadraticProblem:
        if y is None:
            raise ValueError("TODO: missing second data")
        kwargs.pop("xx", None)
        kwargs.pop("yy", None)
        # TODO(michalk8): pass epsilon
        geom_x = self._create_geometry(x, **kwargs)
        geom_y = self._create_geometry(y, **kwargs)

        # TODO(michalk8): marginals + kwargs?
        return QuadraticProblem(geom_x, geom_y, geom_xy=None, fused_penalty=0.0, a=a, b=b, tau_a=tau_a, tau_b=tau_b)

    @property
    def _solver_output(self) -> Tuple[Type[Any], Type[BaseSolverOutput]]:
        return GW, GWOutput


class FGWSolver(GWSolver):
    def _prepare_input(
        self,
        x: TaggedArray,
        y: Optional[TaggedArray] = None,
        a: Optional[npt.ArrayLike] = None,
        b: Optional[npt.ArrayLike] = None,
        tau_a: Optional[float] = 1.0,
        tau_b: Optional[float] = 1.0,
        xx: Optional[TaggedArray] = None,
        yy: Optional[TaggedArray] = None,
        **kwargs: Any,
    ) -> QuadraticProblem:
        if xx is None:
            raise ValueError("TODO: no array defining joint")
        problem = super()._prepare_input(x, y, **kwargs)

        if yy is None:
            if not xx.is_cost_matrix and not xx.is_kernel:
                raise ValueError("TODO")
            geom_xy = self._create_geometry(xx)
        else:
            geom_xy = self._create_geometry(xx, yy)
        self._validate_geoms(problem.geom_xx, problem.geom_yy, geom_xy)

        # TODO(michalk8): marginals + kwargs?
        return QuadraticProblem(
            problem.geom_xx,
            problem.geom_yy,
            geom_xy,
            fused_penalty=0.5,
            a=a,
            b=b,
            tau_a=tau_a,
            tau_b=tau_b,
        )

    @staticmethod
    def _validate_geoms(geom_x: Geometry, geom_y: Geometry, geom_xy: Geometry) -> None:
        # TODO(michalk8): check if this is right
        if geom_x.shape[0] != geom_xy.shape[0]:
            raise ValueError("TODO: first and joint geom mismatch")
        if geom_y.shape[0] != geom_xy.shape[1]:
            raise ValueError("TODO: second and joint geom mismatch")

    def _solve(self, data: QuadraticProblem, alpha: float = 0.5, **kwargs: Any) -> GWOutput:
        if not (0 < alpha < 1):
            raise ValueError("TODO: wrong alpha range")
        if alpha != data.fused_penalty:
            data.fused_penalty = alpha

        return super()._solve(data, **kwargs)
