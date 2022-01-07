# TODO: This file should be independent of backend, e.g. JAX
from abc import abstractmethod
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from sklearn.base import BaseEstimator
from anndata import AnnData
from moscot.framework.results._result_mixins import ResultMixin



class BaseProblem(BaseEstimator, ResultMixin):
    """Base estimator for OT problems."""

    def __init__(
        self,
        adata: AnnData = None,
    ) -> None:
        self._adata = adata

    @abstractmethod
    def prepare(
        self
    ) -> "BaseProblem":
        pass

    @abstractmethod
    def solve(
        self,
        inplace: bool = True
    ) -> Optional["BaseResult"]:
        pass

    @abstractmethod
    def prepare(
            self
    ) -> "BaseProblem":
        pass

    @property
    def adata(self) -> AnnData:
        return self._adata

    @property
    @abstractmethod
    def solvers(self) -> Dict[Tuple, Any]:
        pass

    @property
    @abstractmethod
    def transport_sets(self) -> List[Tuple]:
        pass

    @property
    @abstractmethod
    def transport_matrix(self) -> Dict[Tuple, np.ndarray]:
        pass




