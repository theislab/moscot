from typing import Any, Optional

from ott.geometry import Grid, Geometry, PointCloud
from ott.core.problems import LinearProblem, QuadraticProblem
from ott.core.sinkhorn import make as Sinkhorn
from ott.core.gromov_wasserstein import make as GW
import jax.numpy as jnp
import numpy.typing as npt

from moscot.tmp.solvers._data import TaggedArray
from moscot.tmp.backends.ott._output import GWOutput, SinkhornOutput
from moscot.tmp.solvers._base_solver import BaseSolver, SolverInput

# TODO(michalk8): initialize ott solvers in init (so that they are not re-jitted


class GeometryMixin:
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
            x = ensure_2D(x.data)
            y = ensure_2D(y.data)
            if x.shape[1] != y.shape[1]:
                raise ValueError("TODO: x/y dimension mismatch")
            return PointCloud(x, y=y, epsilon=eps, **kwargs)
        if x.is_point_cloud:
            return PointCloud(ensure_2D(x.data), epsilon=eps, **kwargs)
        if x.is_grid:
            return Grid(jnp.asarray(x.data), epsilon=eps, **kwargs)
        if x.is_cost:
            return Geometry(cost_matrix=ensure_2D(x.data, allow_reshape=False), epsilon=eps, **kwargs)
        if x.is_kernel:
            return Geometry(kernel_matrix=ensure_2D(x.data, allow_reshape=False), epsilon=eps, **kwargs)

        raise NotImplementedError(x)


class SinkhornSolver(GeometryMixin, BaseSolver):
    def __init__(self, **kwargs: Any):
        super().__init__()
        self._solver = Sinkhorn(**kwargs)

    def _prepare_input(
        self,
        x: TaggedArray,
        y: Optional[TaggedArray] = None,
        **kwargs: Any,
    ) -> SolverInput:
        if y is not None:
            return SolverInput(xy=self._create_geometry(x, y, **kwargs))
        return SolverInput(xy=self._create_geometry(x, **kwargs))

    def _solve(self, data: SolverInput, **kwargs: Any) -> SinkhornOutput:
        # TODO(michalk8): marginals
        # TODO(michalk8): add method to modify eps (fairly simple) and allow to modify here?
        problem = LinearProblem(data.xy)
        return SinkhornOutput(self._solver(problem))


class GWSolver(GeometryMixin, BaseSolver):
    def __init__(self, **kwargs: Any):
        super().__init__()
        self._solver = GW(**kwargs)
        self._alpha = 0.0

    def _prepare_input(
        self,
        x: TaggedArray,
        y: Optional[TaggedArray] = None,
        **kwargs: Any,
    ) -> SolverInput:
        if y is None:
            raise ValueError("TODO: missing second data")
        # TODO(michalk8): pass epsilon
        geom_x = self._create_geometry(x, **kwargs)
        geom_y = self._create_geometry(y, **kwargs)

        # TODO(michalk8): marginals
        return SolverInput(x=geom_x, y=geom_y)

    def _solve(self, data: SolverInput, **kwargs: Any) -> GWOutput:
        # TODO(michalk8): marginals + loss
        problem = QuadraticProblem(data.x, data.y, data.xy, fused_penalty=self._alpha, is_fused=self._alpha > 0)
        return GWOutput(self._solver(problem))


class FGWSolver(GWSolver):
    def __init__(self, alpha: float = 0.5, **kwargs: Any):
        if not (0 < alpha < 1):
            raise ValueError()
        super().__init__(**kwargs)
        self._alpha = alpha

    @staticmethod
    def _validate_geoms(geom_x: Geometry, geom_y: Geometry, geom_xy: Geometry) -> None:
        # TODO(michalk8): check if this is right
        if geom_x.shape[0] != geom_xy.shape[0]:
            raise ValueError("TODO: first and joint geom mismatch")
        if geom_y.shape[0] != geom_xy.shape[1]:
            raise ValueError("TODO: second and joint geom mismatch")

    def _prepare_input(
        self,
        x: TaggedArray,
        y: Optional[TaggedArray] = None,
        xx: Optional[TaggedArray] = None,
        yy: Optional[TaggedArray] = None,
        **kwargs: Any,
    ) -> SolverInput:
        if xx is None:
            raise ValueError("TODO: no array defining joint")
        res = super()._prepare_input(x, y, **kwargs)

        if yy is None:
            if not xx.is_cost and not xx.is_kernel:
                raise ValueError("TODO")
            geom_xy = self._create_geometry(xx)
        else:
            geom_xy = self._create_geometry(xx, yy)
        self._validate_geoms(res.x, res.y, geom_xy)

        # TODO(michalk8): marginals
        return SolverInput(x=res.x, y=res.y, xy=geom_xy)
