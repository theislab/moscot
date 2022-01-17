from abc import ABC, abstractmethod
from typing import Any, Dict, Union, Optional

from typing_extensions import Literal

from jax import numpy as jnp
from ott.geometry.costs import CostFn
from ott.geometry.geometry import Geometry

GWLoss = type(None)
import numpy as np

CostFn_t = Union[CostFn, GWLoss]


class BaseSolver(ABC):
    """Base solver for OT problems."""

    def __init__(self, cost_fn: Optional[CostFn_t] = None):
        self._cost_fn = cost_fn or self._default_cost_fn

    @property
    @abstractmethod
    def _default_cost_fn(self) -> Union[CostFn, GWLoss]:
        pass

    # TODO(michalk8): in the future, let `_fit` be abstract and `fit` call `_check_marginals` in the end
    @abstractmethod
    def fit(
        self,
        geom: Union[jnp.array, Geometry],
        a: Optional[jnp.array] = None,
        b: Optional[jnp.array] = None,
        **kwargs: Any,
    ) -> "BaseSolver":
        """TODO."""

    # TODO(michalk8): in the future, consider exposing the tolerances e.g. via persistent config
    def _check_marginals(
        self, a: Optional[jnp.array] = None, b: Optional[jnp.array] = None, rtol: float = 1e-2, atol: float = 1e-2
    ) -> None:
        def assert_isclose(expected_marginals: Optional[jnp.ndarray], *, axis: Literal[0, 1]) -> None:
            matrix = self.matrix
            if expected_marginals is None:
                expected_marginals = jnp.ones((matrix.shape[axis],)) / matrix.shape[axis]

            try:
                np.testing.assert_allclose(
                    matrix.sum(1 - axis),
                    expected_marginals,
                    rtol=rtol,
                    atol=atol,
                    verbose=False,
                    err_msg=f"{'Target' if axis else 'Source'} marginals do not match the expected marginals.",
                )
            except AssertionError as e:
                raise ValueError(str(e)) from None

        assert_isclose(a, axis=0)
        assert_isclose(b, axis=1)

    # TODO(michalk8): add predict (alias for transport?)
    # TODO(michalk8): add some basic visualization (optional matplotlib req)
    @abstractmethod
    def transport(self, inputs: jnp.ndarray, forward: bool = True) -> jnp.ndarray:
        """Transport mass."""

    @property
    @abstractmethod
    def converged(self) -> Optional[bool]:
        """`True` if the solver converged."""

    @property
    @abstractmethod
    def matrix(self) -> jnp.ndarray:
        """Transport matrix."""

    @property
    def params(self) -> Dict[str, Any]:
        """NYI: Solver parameters."""
        return NotImplemented
        # return self.get_params(deep=True)
