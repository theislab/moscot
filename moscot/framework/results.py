from typing import Any, Dict, List, Tuple, Union, Literal, Optional
from numbers import Number

from networkx import DiGraph
from abc import ABC, abstractmethod

from jax import numpy as jnp
from ott.geometry.costs import CostFn
from ott.geometry.geometry import Geometry
from ott.core.gromov_wasserstein import GWLoss
import numpy as np
from anndata import AnnData

CostFn_t = Union[CostFn, GWLoss]



class BaseResult(ABC):
    """Base result for OT problems."""

    def __init__(self,
                 adata: AnnData,
                 estimators_dict: Dict[Tuple, 'BaseProblem']) -> None:
        self._adata = adata
        self._estimators = estimators_dict
        self._groups = None

    def group_cells(self, key: Union[str, List[str]]) -> List:
        if isinstance(key, str):
            adata_groups = self._adata.obs[key].values
            assert key in adata_groups
            self._groups = list(adata_groups)
        else:
            self._groups = key

    @property
    def adata(self) -> AnnData:
        return self._adata

    @property
    def estimators(self) -> Union[List['BaseProblem'], 'BaseProblem']:
        return self._estimators


class OTResult(BaseResult):
    def __init__(self, adata: AnnData,
                 estimators_dict: Dict[Tuple, 'BaseProblem']) -> None:
        self.matrix_dict = {tup: getattr(estimator, "matrix") for tup, estimator in estimators_dict.items()}
        super().__init__(adata=adata, estimators=estimators_dict)

    def plot_aggregate_transport(self, key: str, groups: List[str]):
        pass

    def push_forward(self,
                     start,
                     end,
                     key,
                     groups):
        pass

    def pull_backward(self, start, end, keys, groups):
        pass