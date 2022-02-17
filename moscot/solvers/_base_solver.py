from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Any, Union, Mapping, Optional
import warnings

import numpy.typing as npt

from moscot.solvers._utils import _warn_not_close
from moscot.solvers._output import BaseSolverOutput
from moscot.solvers._tagged_array import Tag, TaggedArray

__all__ = ("BaseSolver",)

ArrayLike = Union[npt.ArrayLike, TaggedArray]


class BaseSolver(ABC):
    @abstractmethod
    def _prepare_input(
        self,
        x: TaggedArray,
        y: Optional[TaggedArray] = None,
        eps: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        pass

    @abstractmethod
    def _solve(self, data: Any, **kwargs: Any) -> BaseSolverOutput:
        pass

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
        eps: Optional[float] = None,
        solve_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **prepare_kwargs: Any,
    ) -> BaseSolverOutput:
        # currently we don't provide x_tag as kwarg, hence we would always convert tags here
        if not isinstance(x, TaggedArray):
            x = _to_tagged_array(x, prepare_kwargs.pop("x_tag", Tag.POINT_CLOUD))
        if not isinstance(y, TaggedArray):
            y = _to_tagged_array(y, prepare_kwargs.pop("y_tag", Tag.POINT_CLOUD))
        if not isinstance(xx, TaggedArray):
            xx = _to_tagged_array(xx, prepare_kwargs.pop("xx_tag", Tag.COST_MATRIX) if yy is None else Tag.POINT_CLOUD)
        if not isinstance(yy, TaggedArray):
            yy = _to_tagged_array(yy, prepare_kwargs.pop("yy_tag", Tag.POINT_CLOUD))

        data = self._prepare_input(x, y, eps=eps, xx=xx, yy=yy, a=a, b=b, tau_a=tau_a, tau_b=tau_b, **prepare_kwargs)
        res = self._solve(data, **solve_kwargs)

        if not res.converged:
            warnings.warn("Solver did not converge")

        n, m = res.shape
        if tau_a == 1.0:
            _warn_not_close((res._ones(n) / n) if a is None else a, res.a, kind="source")
        if tau_b == 1.0:
            _warn_not_close((res._ones(m) / m) if b is None else b, res.b, kind="target")

        return res


def _to_tagged_array(arr: Optional[ArrayLike], tag: Tag) -> Optional[TaggedArray]:
    if arr is None:
        return None

    tag = Tag(tag)
    if not isinstance(arr, TaggedArray):
        return TaggedArray(arr, tag=tag)

    return TaggedArray(arr.data, tag=tag)
