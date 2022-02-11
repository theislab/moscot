from abc import ABC, abstractmethod
from typing import Any, Union, Optional

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
        a: npt.ArrayLike = None,
        b: npt.ArrayLike = None,
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
        **kwargs: Any,
    ) -> BaseSolverOutput:
        def to_tagged_array(arr: Optional[ArrayLike], tag: Tag) -> Optional[TaggedArray]:
            if arr is None:
                return None
            tag = Tag(tag)
            if not isinstance(arr, TaggedArray):
                return TaggedArray(arr, tag=tag)
            # force new tag
            return TaggedArray(arr.data, tag=tag)

        # currently we don't provide x_tag as kwarg, hence we would always convert tags here
        if not isinstance(x, TaggedArray):
            x = to_tagged_array(x, kwargs.pop("x_tag", Tag.POINT_CLOUD))
        if not isinstance(y, TaggedArray):
            y = to_tagged_array(y, kwargs.pop("y_tag", Tag.POINT_CLOUD))
        if not isinstance(xx, TaggedArray):
            xx = to_tagged_array(xx, kwargs.pop("xx_tag", Tag.COST_MATRIX) if yy is None else Tag.POINT_CLOUD)
        if not isinstance(yy, TaggedArray):
            yy = to_tagged_array(yy, kwargs.pop("yy_tag", Tag.POINT_CLOUD))

        # TODO(michalk8): create TaggedArray here if not passed, taking x_tag/y_tag/xy_tag from kwargs
        # TODO(michak8): filter kwargs
        data = self._prepare_input(x, y, a, b, xx=xx, yy=yy, tau_a=tau_a, tau_b=tau_b, eps=eps)
        res = self._solve(data, **kwargs)

        n, m = res.shape
        # TODO(michalk8): how to handle unbalanced?
        _warn_not_close((res._ones(n) / n) if a is None else a, res.a, kind="source")
        _warn_not_close((res._ones(m) / m) if b is None else b, res.b, kind="target")

        return res
