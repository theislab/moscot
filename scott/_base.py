from abc import ABC, abstractmethod
from typing import Any, Dict, Union, Optional

from jax import numpy as jnp
from ott.geometry.geometry import Geometry


class BaseCostFn(ABC):
    @abstractmethod
    def __call__(self, x: jnp.array, y: jnp.array):
        pass


class BaseSolver(ABC):
    """TODO."""

    def __init__(self, cost_fn: BaseCostFn):
        self._cost_fn = cost_fn

    @property
    @abstractmethod
    def matrix(self) -> jnp.ndarray:
        """TODO."""

    @property
    def params(self) -> Dict[str, Any]:
        """TODO."""
        return NotImplemented
        # return self.get_params(deep=True)

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
