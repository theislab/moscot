from typing import Any

from numpy import typing as npt

from moscot.tmp.mixins._base_analysis import AnalysisMixin


class TemporalAnalysisMixin(AnalysisMixin):
    def push_forward(self, *args: Any, **kwargs: Any) -> npt.ArrayLike:
        pass

    def pull_backward(self, *args: Any, **kwargs: Any) -> npt.ArrayLike:
        pass
