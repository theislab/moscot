from abc import ABC, abstractmethod
from typing import Any

import numpy.typing as npt


class AnalysisMixin(ABC):
    @abstractmethod
    def push_forward(self, *args: Any, **kwargs: Any) -> npt.ArrayLike:
        pass

    @abstractmethod
    def pull_backward(self, *args: Any, **kwargs: Any) -> npt.ArrayLike:
        pass
