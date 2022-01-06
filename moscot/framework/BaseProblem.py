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
    def prepare(
        self,
        key: Union[str, None],
        policy: None,
        rep: None,
        cost_fn: Union[CostFn, None],
        eps: Union[float, None],
        groups: Union[List[str], Tuple[str]],
    ) -> "BaseProblem":
        pass

    @abstractmethod
    def fit(
        self,
        epsilon: Optional[Union[List[Union[float, Epsilon]], float, Epsilon]] = 0.5,
        tau_a: Optional[Union[List, Dict[Tuple, List], float]] = 1.0,
        tau_b: Optional[Union[List, Dict[Tuple, List], float]] = 1.0,
    ) -> "BaseResult":
        pass

    @property
    def adata(self) -> AnnData:
        return self._adata

    @property
    def rep(self) -> str:
        return self._rep

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
    def transport_matrix(self) -> Dict[Tuple, jnp.ndarray]:
        pass




