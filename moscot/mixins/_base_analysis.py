from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from sklearn.linear_model import LinearRegression
from anndata import AnnData
import numpy.typing as npt


# TODO(michalk8): need to think about this a bit more
class AnalysisMixin(ABC):
    @abstractmethod
    def push_forward(self, *args: Any, **kwargs: Any) -> npt.ArrayLike:
        pass

    @abstractmethod
    def pull_backward(self, *args: Any, **kwargs: Any) -> npt.ArrayLike:
        pass


# TODO(michalk8): CompoundAnalysisMixin?

class CompoundAnalysisMixin(AnalysisMixin):

    @staticmethod
    def _project_on_distribution(adata_source: AnnData, # TODO: Make low-rank
                                 adata_new: AnnData,
                                 n_jobs = 1) -> npt.ArrayLike:
        reg = LinearRegression(fit_intercept=False, n_jobs=n_jobs).fit(adata_source.X.T, adata_new.X)  # (n_genes, n_cells) and (n_genes,1)
        return reg.coef_

