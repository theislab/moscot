from abc import ABC
from typing import List, Union

from ott.geometry.costs import CostFn
from ott.core.gromov_wasserstein import GWLoss

CostFn_t = Union[CostFn, GWLoss]
from anndata import AnnData
from moscot.framework.estimators import BaseProblem


class BaseResult(ABC):
    """Base result for OT problems."""

    def __init__(self,
                 adata: AnnData,
                 _estimators: Union[List[BaseProblem], BaseProblem]) -> None:
        self.adata = adata
        self._estimators = _estimators

    def group_by(self, key: str):
        pass


class OTResult(BaseResult):
    def __init__(self,
                 adata: AnnData,
                 _estimators: Union[List[BaseProblem], BaseProblem]
                ) -> None:
        super().__init__(adata=adata, _estimators=_estimators)

        return self

    def plot_aggregate_transport(self, key: str, groups: List[str]):
        pass

    def push_forward(self, start, end, key, groups):
        pass

    def pull_backward(self, start, end, keys, groups):
        pass
