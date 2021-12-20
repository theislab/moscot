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

    def __init__(self, adata: AnnData, _estimators: Union[List['BaseProblem'], 'BaseProblem']) -> None:
        self.adata = adata
        self._estimators = _estimators

    def group_by(self, key: str):
        pass


class OTResult(BaseResult):
    def __init__(self, adata: AnnData, _estimators: Union[List['BaseProblem'], 'BaseProblem']) -> None:
        super().__init__(adata=adata, _estimators=_estimators)


    def plot_aggregate_transport(self, key: str, groups: List[str]):
        pass

    def push_forward(self, start, end, key, groups):
        pass

    def pull_backward(self, start, end, keys, groups):
        pass