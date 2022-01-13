from abc import abstractmethod
from typing import Any, Type, Union, Optional

from ott.geometry import Grid, Geometry, PointCloud
from ott.core.problems import LinearProblem, QuadraticProblem
from ott.core.sinkhorn import make as Sinkhorn
from ott.geometry.costs import Euclidean
from ott.core.gromov_wasserstein import make as GW
import jax.numpy as jnp
import numpy.typing as npt

from moscot.tmp.solvers._data import TaggedArray

# TODO(michalk8): initialize ott solvers in init (so that they are not re-jitted
from moscot.tmp.solvers._output import BaseSolverOutput
from moscot.tmp.backends.ott._output import GWOutput, SinkhornOutput, LRSinkhornOutput
from moscot.tmp.solvers._base_solver import BaseSolver


class GeometryMixin:
    @property
    @abstractmethod
    def _output_type(self) -> Type[BaseSolverOutput]:
        pass

    def _create_geometry(
        self,
        x: TaggedArray,
        y: Optional[TaggedArray] = None,
        *,
        eps: Optional[float] = None,
        **kwargs: Any,
    ) -> Geometry:
        def ensure_2D(arr: npt.ArrayLike, *, allow_reshape: bool = True) -> jnp.ndarray:
            arr = jnp.asarray(arr)
            arr = jnp.reshape(arr, (-1, 1)) if (allow_reshape and arr.ndim == 1) else arr
            if arr.ndim != 2:
                raise ValueError("TODO: expected 2D")
            return arr

        if y is not None:
            cost_fn = x.loss  # TODO: need mapping from string to ott cost_fn
            x = ensure_2D(x.data)
            y = ensure_2D(y.data)
            if x.shape[1] != y.shape[1]:
                raise ValueError("TODO: x/y dimension mismatch")
            return PointCloud(x, y=y, epsilon=eps, cost_fn=cost_fn, **kwargs)
        if x.is_point_cloud:
            print(x.loss)
            cost_fn = x.loss  # TODO: need mapping from string to ott cost_fn
            return PointCloud(ensure_2D(x.data), epsilon=eps, cost_fn=cost_fn, **kwargs)
        if x.is_grid:
            cost_fn = kwargs.pop("cost_fn", Euclidean())
            return Grid(jnp.asarray(x.data), epsilon=eps, cost_fn=cost_fn, **kwargs)
        if x.is_cost_matrix:
            return Geometry(
                cost_matrix=ensure_2D(x.loss, allow_reshape=False), epsilon=eps, **kwargs
            )  # TODO do also want to have cost matrices saved in x.data? Now changed it to x.loss, discuss
        if x.is_kernel:
            return Geometry(kernel_matrix=ensure_2D(x.loss, allow_reshape=False), epsilon=eps, **kwargs)

        raise NotImplementedError(x)

    def _set_eps(
        self, problem: Union[LinearProblem, QuadraticProblem], eps: Optional[float] = None
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

    def _solve(self, data: Union[LinearProblem, QuadraticProblem], **kwargs: Any) -> BaseSolverOutput:
        return self._output_type(self._solver(data, **kwargs))


class SinkhornSolver(GeometryMixin, BaseSolver):
    def __init__(self, **kwargs: Any):
        super().__init__()
        self._solver = Sinkhorn(**kwargs)

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
        geom = self._create_geometry(x, **kwargs) if y is None else self._create_geometry(x, y, **kwargs)
        return LinearProblem(geom, a=a, b=b, tau_a=tau_a, tau_b=tau_b)

    @property
    def _output_type(self) -> Type[BaseSolverOutput]:
        return SinkhornOutput


class LRSinkhorn(SinkhornSolver):
    @property
    def _output_type(self) -> Type[BaseSolverOutput]:
        return LRSinkhornOutput

    def _solve(self, data: Union[LinearProblem, QuadraticProblem], **kwargs: Any) -> BaseSolverOutput:
        return self._output_type(self._solver(data, **kwargs), converged=self._solver.threshold)


class GWSolver(GeometryMixin, BaseSolver):
    def __init__(self, **kwargs: Any):
        super().__init__()
        self._solver = GW(**kwargs)

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
        return QuadraticProblem(
            geom_x, geom_y, geom_xy=None, fused_penalty=0.0, a=a, b=b, is_fused=False, tau_a=tau_a, tau_b=tau_b
        )

    @property
    def _output_type(self) -> Type[BaseSolverOutput]:
        return GWOutput


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
            is_fused=False,
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
