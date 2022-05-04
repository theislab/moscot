from abc import ABC, abstractmethod
from enum import Enum
from types import MappingProxyType
from typing import Any, Type, Tuple, Union, Literal, Mapping, Optional, NamedTuple
import warnings

import numpy.typing as npt

from moscot._docs import d
from moscot.solvers._utils import _warn_not_close
from moscot.solvers._output import BaseSolverOutput
from moscot.solvers._tagged_array import Tag, TaggedArray

__all__ = ("ProblemKind", "BaseSolver", "OTSolver")

# TODO(michalk8): consider making TaggedArray private (used only internally)?
ArrayLike = Union[npt.ArrayLike, TaggedArray]


class ProblemKind(str, Enum):
    """Class defining the problem class and dispatching the solvers."""
    LINEAR = "linear"
    QUAD = "quadratic"
    QUAD_FUSED = "quadratic_fused"

    def solver(self, *, backend: Literal["ott"] = "ott") -> Type["BaseSolver"]:
        """
        Return the solver dependent on the backend and the problem type.

        Parameters
        ----------
        backend
            The backend the solver corresponds to.

        Returns
        -------
        Solver
            Solver corresponding to the backend and problem type.
        """
        if backend == "ott":
            from moscot.backends.ott import GWSolver, FGWSolver, SinkhornSolver

            if self == ProblemKind.LINEAR:
                return SinkhornSolver
            if self == ProblemKind.QUAD:
                return GWSolver
            if self == ProblemKind.QUAD_FUSED:
                return FGWSolver
            raise NotImplementedError(self)

        raise NotImplementedError(f"Invalid backend: `{backend}`")


class TaggedArrayData(NamedTuple):
    x: Optional[TaggedArray]
    y: Optional[TaggedArray]
    xy: Tuple[Optional[TaggedArray], Optional[TaggedArray]]


class TagConverterMixin:
    def _get_array_data(
        self,
        xy: Optional[Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]] = None,
        x: Optional[ArrayLike] = None,
        y: Optional[ArrayLike] = None,
        tags: Mapping[Literal["xy", "x", "y"], Tag] = MappingProxyType({}),
    ) -> TaggedArrayData:
        x, y = self._convert(x, y, tags=tags, is_linear=False)
        if xy is None:
            xy = (None, None)
        elif not isinstance(xy, tuple):
            xy = (xy, None)
        xy = self._convert(xy[0], xy[1], tags=tags, is_linear=True)
        return TaggedArrayData(x=x, y=y, xy=xy)

    @staticmethod
    def _convert(
        x: Optional[ArrayLike],
        y: Optional[ArrayLike],
        tags: Mapping[Literal["xy", "x", "y"], Tag] = MappingProxyType({}),
        *,
        is_linear: bool,
    ) -> Tuple[Optional[TaggedArray], Optional[TaggedArray]]:
        def to_tagged_array(arr: Optional[ArrayLike], tag: Tag) -> Optional[TaggedArray]:
            if arr is None:
                return None
            tag = Tag(tag)
            if isinstance(arr, TaggedArray):
                return TaggedArray(arr.data, tag=tag)
            return TaggedArray(arr, tag=tag)

        def cost_or_kernel(arr: TaggedArray, key: Literal["xy", "x", "y"]) -> TaggedArray:
            arr = to_tagged_array(arr, tag=tags.get(key, Tag.COST_MATRIX))
            if arr.tag not in (Tag.COST_MATRIX, Tag.KERNEL):
                raise ValueError(f"TODO: wrong tag - expected kernel/cost, got `{arr.tag}`")
            return arr

        x_key, y_key = ("xy", "xy") if is_linear else ("x", "y")
        if x is None and y is None:
            return None, None  # checks are done later
        if x is None:
            return cost_or_kernel(y, key=y_key), None
        if y is None:
            return cost_or_kernel(x, key=x_key), None
        if is_linear:
            return to_tagged_array(x, tag=Tag.POINT_CLOUD), to_tagged_array(y, tag=Tag.POINT_CLOUD)
        return to_tagged_array(x, tag=tags.get(x_key, Tag.POINT_CLOUD)), to_tagged_array(
            y, tag=tags.get(y_key, Tag.POINT_CLOUD)
        )


class BaseSolver(ABC):
    """BaseSolver class."""

    @abstractmethod
    def _prepare(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        pass

    @abstractmethod
    def _solve(self, data: Any, **kwargs: Any) -> BaseSolverOutput:
        pass

    @property
    @abstractmethod
    def problem_kind(self) -> ProblemKind:
        """Problem kind."""
        # helps to check whether necessary inputs were passed

    def __call__(
        self,
        *args: Any,
        solve_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> BaseSolverOutput:
        """Call method."""
        data = self._prepare(*args, **kwargs)
        return self._solve(data, **solve_kwargs)


@d.get_sections(base="OTSolver", sections=["Parameters", "Raises"])
class OTSolver(TagConverterMixin, BaseSolver, ABC):
    """OTSolver class."""

    def __call__(
        self,
        xy: Optional[Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]] = None,
        x: Optional[ArrayLike] = None,
        y: Optional[ArrayLike] = None,
        a: Optional[npt.ArrayLike] = None,
        b: Optional[npt.ArrayLike] = None,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        tags: Mapping[Literal["x", "y", "xy"], Tag] = MappingProxyType({}),
        solve_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> BaseSolverOutput:
        """Call method."""
        data = self._get_array_data(xy, x=x, y=y, tags=tags)
        kwargs = self._prepare_kwargs(data, **kwargs)

        res = super().__call__(a=a, b=b, tau_a=tau_a, tau_b=tau_b, solve_kwargs=solve_kwargs, **kwargs)

        return self._check_marginals(res, a=a, b=b, tau_a=tau_a, tau_b=tau_b)

    def _prepare_kwargs(
        self,
        data: TaggedArrayData,
        **kwargs: Any,
    ) -> Mapping[str, Union[Optional[TaggedArray], Any]]:
        def assert_linear() -> None:
            if data.xy == (None, None):
                raise ValueError("TODO: no linear data.")

        def assert_quadratic() -> None:
            if data.x is None or data.y is None:
                raise ValueError("TODO: no quadratic data.")

        if self.problem_kind == ProblemKind.LINEAR:
            assert_linear()
            data_kwargs = {"xy": data.xy}
        elif self.problem_kind == ProblemKind.QUAD:
            assert_quadratic()
            data_kwargs = {"x": data.x, "y": data.y}
        elif self.problem_kind == ProblemKind.QUAD_FUSED:
            assert_linear()
            assert_quadratic()
            data_kwargs = {"x": data.x, "y": data.y, "xy": data.xy}
        else:
            raise NotImplementedError(f"TODO: {self.problem_kind}")

        if self.problem_kind != ProblemKind.QUAD_FUSED:
            kwargs.pop("alpha", None)

        return {**kwargs, **data_kwargs}

    @staticmethod
    def _check_marginals(
        res: BaseSolverOutput,
        a: Optional[npt.ArrayLike] = None,
        b: Optional[npt.ArrayLike] = None,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
    ) -> BaseSolverOutput:
        if not res.converged:
            warnings.warn("Solver did not converge")
        n, m = res.shape
        tol_source = 1 / (n * 10)  # TODO(giovp): maybe round?
        tol_target = 1 / (m * 10)
        if tau_a == 1.0:
            _warn_not_close(
                (res._ones(n) / n) if a is None else a, res.a, kind="source", rtol=tol_source, atol=tol_source
            )
        if tau_b == 1.0:
            _warn_not_close(
                (res._ones(m) / m) if b is None else b, res.b, kind="target", rtol=tol_target, atol=tol_target
            )

        return res
