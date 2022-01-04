from abc import abstractmethod
from typing import Any, Dict, List, Tuple, Union, Optional
from numbers import Number
from jax import numpy as jnp
from ott.geometry.costs import CostFn
from ott.geometry.geometry import Geometry
from ott.core.gromov_wasserstein import GWLoss

CostFn_t = Union[CostFn, GWLoss]
from sklearn.base import BaseEstimator

from ott.geometry.costs import Euclidean
from ott.geometry.epsilon_scheduler import Epsilon

from anndata import AnnData



class BaseProblem(BaseEstimator):
    """Base estimator for OT problems."""

    def __init__(
        self,
        adata: AnnData = None,
        rep: str = "X",
    ) -> None:
        self._adata = adata
        self._rep = rep

    @abstractmethod
    def serialize_to_adata(self) -> Optional[AnnData]:
        pass

    @abstractmethod
    def load_from_adata(self) -> None:
        pass

    @abstractmethod
    def prepare(
        self,
        key: Union[str, None],
        policy: None,
        rep: None,
        cost_fn: Union[CostFn, None],
        eps: Union[float, None],
        groups: Union[List[str], Tuple[str]],
    ) -> None:
        pass

    @abstractmethod
    def fit(
        self,
        geom: Union[jnp.array, Geometry],
        a: Optional[jnp.array] = None,
        b: Optional[jnp.array] = None,
    ) -> 'BaseResult':
        pass

    @abstractmethod
    def _create_cost(self, cost: Union[CostFn, None] = Euclidean) -> None:
        pass

    @property
    def adata(self) -> AnnData:
        return self._adata

    @property
    def rep(self) -> str:
        return self._rep




