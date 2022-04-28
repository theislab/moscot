from abc import ABC, abstractmethod
from enum import auto, Enum
from types import MappingProxyType
from typing import Any, Tuple, Union, Literal, Mapping, Optional, NamedTuple
import warnings

import numpy.typing as npt

from moscot.solvers._utils import _warn_not_close
from moscot.solvers._output import BaseSolverOutput
from moscot.solvers._tagged_array import Tag, TaggedArray

__all__ = ("BaseSolver",)

ArrayLike = Union[npt.ArrayLike, TaggedArray]


class ProblemKind(Enum):
    LINEAR = auto()
    QUAD = auto()
    QUAD_FUSED = auto()


class ArrayData(NamedTuple):
    x: TaggedArray
    y: TaggedArray
    xy: TaggedArray


class TagConverterMixin:
    def _convert(
        self,
        x: ArrayLike,
        y: Optional[ArrayLike] = None,
        xy: Optional[Tuple[ArrayLike, ArrayLike]] = None,
        tags: Mapping[Literal["x", "y", "xy"], Tag] = MappingProxyType({}),
    ) -> ArrayData:
        if not isinstance(x, TaggedArray):
            x = self._to_tagged_array(x, tags.get("x", Tag.POINT_CLOUD))
        if not isinstance(y, TaggedArray):
            y = self._to_tagged_array(y, tags.get("y", Tag.POINT_CLOUD))

        if xy is None:
            return x, y, None

        xx, yy = xy
        if yy is None:
            xx = self._to_tagged_array(xx, tag=tags.get("xy", Tag.COST_MATRIX))
            return ArrayData(x, y, (xx, None))

        xx = self._to_tagged_array(xx, tag=Tag.POINT_CLOUD)
        yy = self._to_tagged_array(yy, tag=Tag.POINT_CLOUD)

        return ArrayData(x, y, (xx, yy))

    @staticmethod
    def _to_tagged_array(arr: Optional[ArrayLike], tag: Tag) -> Optional[TaggedArray]:
        if arr is None:
            return None
        tag = Tag(tag)
        if isinstance(arr, TaggedArray):
            return TaggedArray(arr.data, tag=tag)
        return TaggedArray(arr, tag=tag)


class BaseSolver(TagConverterMixin, ABC):
    """BaseSolver class."""

    @abstractmethod
    def _prepare_input(
        self,
        x: TaggedArray,
        y: Optional[TaggedArray] = None,
        epsilon: Optional[float] = None,
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
        x: Optional[ArrayLike] = None,
        y: Optional[ArrayLike] = None,
        xy: Optional[Tuple[TaggedArray, TaggedArray]] = None,
        a: Optional[npt.ArrayLike] = None,
        b: Optional[npt.ArrayLike] = None,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        epsilon: Optional[float] = None,
        tags: Mapping[Literal["x", "y", "xy"], Tag] = MappingProxyType({}),
        solve_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> BaseSolverOutput:
        """Call method."""
        data = self._convert(x, y, xy=xy, tags=tags)
        kwargs = self._prepare_kwargs(data, **kwargs)
        data = self._prepare_input(epsilon=epsilon, a=a, b=b, tau_a=tau_a, tau_b=tau_b, **kwargs)
        res = self._solve(data, **solve_kwargs)

        return self._verify_result(res, a=a, b=b, tau_a=tau_a, tau_b=tau_b)

    def _prepare_kwargs(
        self,
        data: ArrayData,
        **kwargs: Any,
    ) -> Mapping[str, Optional[TaggedArray]]:
        if self.problem_kind == ProblemKind.LINEAR:
            data_kwargs = {"x": data.x, "y": data.y}
        elif self.problem_kind == ProblemKind.QUAD:
            data_kwargs = {"xy": data.xy}
        elif self.problem_kind == ProblemKind.QUAD_FUSED:
            data_kwargs = {"x": data.x, "y": data.y, "xy": data.xy}
        else:
            raise NotImplementedError(f"TODO: {self.problem_kind}")

        if self.problem_kind != ProblemKind.QUAD_FUSED:
            kwargs.pop("alpha", None)

        return {**kwargs, **data_kwargs}

    @staticmethod
    def _verify_result(
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
