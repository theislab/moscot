from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Any, Tuple, Union, Literal, Mapping, Optional
from contextlib import contextmanager
import warnings

import numpy.typing as npt

from moscot.solvers._utils import _warn_not_close
from moscot.solvers._output import BaseSolverOutput
from moscot.solvers._tagged_array import Tag, TaggedArray

__all__ = ("BaseSolver", "ContextlessBaseSolver")

ArrayLike = Union[npt.ArrayLike, TaggedArray]


class TagConverterMixin:
    def _convert(
        self,
        x: ArrayLike,
        y: Optional[ArrayLike] = None,
        xx: Optional[ArrayLike] = None,
        yy: Optional[ArrayLike] = None,
        tags: Mapping[Literal["x", "y", "xx", "yy"], Tag] = MappingProxyType({}),
    ) -> Tuple[Optional[TaggedArray], Optional[TaggedArray], Optional[TaggedArray], Optional[TaggedArray]]:
        if not isinstance(x, TaggedArray):
            x = self._to_tagged_array(x, tags.get("x", Tag.POINT_CLOUD))
        if not isinstance(y, TaggedArray):
            y = self._to_tagged_array(y, tags.get("y", Tag.POINT_CLOUD))
        if not isinstance(xx, TaggedArray):
            xx = self._to_tagged_array(xx, tags.get("xx", Tag.COST_MATRIX) if yy is None else Tag.POINT_CLOUD)
        if not isinstance(yy, TaggedArray):
            yy = self._to_tagged_array(yy, tags.get("yy", Tag.POINT_CLOUD))

        return x, y, xx, yy

    @staticmethod
    def _to_tagged_array(arr: Optional[ArrayLike], tag: Tag) -> Optional[TaggedArray]:
        if arr is None:
            return None

        tag = Tag(tag)
        if not isinstance(arr, TaggedArray):
            return TaggedArray(arr, tag=tag)

        return TaggedArray(arr.data, tag=tag)


class BaseSolver(TagConverterMixin, ABC):
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

    @abstractmethod
    def _set_ctx(self, data: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def _reset_ctx(self, old_context: Any) -> None:
        pass

    @contextmanager
    def _solve_ctx(self, data: Any, **kwargs: Any) -> None:
        old_context = self._set_ctx(data, **kwargs)
        try:
            yield
        finally:
            self._reset_ctx(old_context)

    def __call__(
        self,
        x: ArrayLike,
        y: Optional[ArrayLike] = None,
        xx: Optional[ArrayLike] = None,
        yy: Optional[ArrayLike] = None,
        a: Optional[npt.ArrayLike] = None,
        b: Optional[npt.ArrayLike] = None,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        epsilon: Optional[float] = None,
        solve_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **prepare_kwargs: Any,
    ) -> BaseSolverOutput:
        x, y, xx, yy = self._convert(x, y, xx, yy, prepare_kwargs.pop("tags", {}))
        data = self._prepare_input(
            x=x, y=y, epsilon=epsilon, xx=xx, yy=yy, a=a, b=b, tau_a=tau_a, tau_b=tau_b, **prepare_kwargs
        )
        with self._solve_ctx(data, epsilon=epsilon, **prepare_kwargs):
            res = self._solve(data, **solve_kwargs)
        return self._check_result(res, a=a, b=b, tau_a=tau_a, tau_b=tau_b)

    def _check_result(
        self,
        res: BaseSolverOutput,
        a: Optional[npt.ArrayLike] = None,
        b: Optional[npt.ArrayLike] = None,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
    ) -> BaseSolverOutput:
        if not res.converged:
            warnings.warn("Solver did not converge")
        n, m = res.shape
        if tau_a == 1.0:
            _warn_not_close((res._ones(n) / n) if a is None else a, res.a, kind="source")
        if tau_b == 1.0:
            _warn_not_close((res._ones(m) / m) if b is None else b, res.b, kind="target")

        return res


class ContextlessBaseSolver(BaseSolver, ABC):
    def _set_ctx(self, data: Any, **kwargs: Any) -> Any:
        pass

    def _reset_ctx(self, old_context: Any) -> None:
        pass
