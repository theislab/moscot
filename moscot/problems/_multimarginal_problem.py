from abc import ABC, abstractmethod
from typing import Any

import numpy.typing as npt

from anndata import AnnData

from moscot.problems import GeneralProblem

__all__ = ("MultiMarginalProblem",)


class MultiMarginalProblem(GeneralProblem, ABC):
    @abstractmethod
    def _estimate_marginals(self, adata: AnnData, **kwargs: Any) -> npt.ArrayLike:
        pass
