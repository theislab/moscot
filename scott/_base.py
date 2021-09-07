from abc import ABC, abstractmethod
from typing import Any, Dict, Union, Optional

from jax import numpy as jnp
from ott.geometry.costs import CostFn
from ott.geometry.geometry import Geometry
from ott.core.gromov_wasserstein import GWLoss

CostFn_t = Union[CostFn, GWLoss]


class BaseSolver(ABC):
    """TODO."""

    def __init__(self, cost_fn: Optional[CostFn_t] = None):
        self._cost_fn = cost_fn or self._default_cost_fn

    @property
    @abstractmethod
    def _default_cost_fn(self) -> Union[CostFn, GWLoss]:
        pass

    @abstractmethod
    def fit(
        self,
        geom: Union[jnp.array, Geometry],
        a: Optional[jnp.array] = None,
        b: Optional[jnp.array] = None,
        **kwargs: Any,
    ) -> "BaseSolver":
        """TODO."""

    # TODO(michalk8): add predict (alias for transport?)
    # TODO(michalk8): add some basic visualization (optional matplotlib req)
    @abstractmethod
    def transport(self, inputs: jnp.ndarray, forward: bool = True) -> jnp.ndarray:
        pass

    @property
    @abstractmethod
    def matrix(self) -> jnp.ndarray:
        """TODO."""

    @property
    def params(self) -> Dict[str, Any]:
        """TODO."""
        return NotImplemented
        # return self.get_params(deep=True)
