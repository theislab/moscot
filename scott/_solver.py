from abc import ABC
from typing import Any, Dict, Union, Optional

from jax import numpy as jnp
from ott.tools.transport import Transport
from ott.geometry.geometry import Geometry
from ott.geometry.pointcloud import PointCloud
from ott.core.gromov_wasserstein import GWLoss, gromov_wasserstein
from ott.geometry.epsilon_scheduler import Epsilon
import numpy as np

from scott._base import BaseCostFn, BaseSolver
from scott._costs import GWSeuqCostFn
from scott._mixins import GeomMixin, TransportMixin

# TODO(michalk8):


class RegularizedOT(BaseSolver, ABC):
    def __init__(self, cost_fn: Optional[BaseCostFn] = None, epsilon: Optional[Union[float, Epsilon]] = None):
        super().__init__(cost_fn=cost_fn)
        self._epsilon: Epsilon = epsilon if isinstance(epsilon, Epsilon) else Epsilon(target=epsilon)

    @property
    def epsilon(self) -> Epsilon:
        """TODO."""
        return self._epsilon

    # TODO(michalk8): if array, allow it to be kernel/distance matrix
    def _prepare_geom(self, geom: Union[jnp.ndarray, Geometry], **kwargs: Any) -> Geometry:
        if isinstance(geom, np.ndarray):
            geom = jnp.asarray(geom)
        if isinstance(geom, jnp.ndarray):
            geom = PointCloud(geom, cost_fn=self._cost_fn, epsilon=self.epsilon._init, **kwargs)
        if not isinstance(geom, Geometry):
            raise TypeError()

        return geom


class UnbalancedOT(TransportMixin, RegularizedOT):
    def __init__(
        self, cost_fn: Optional[BaseCostFn] = None, epsilon: Optional[Union[float, Epsilon]] = None, **kwargs: Any
    ):
        super().__init__(cost_fn=cost_fn, epsilon=epsilon)
        # TODO(michalk8): check kwargs
        self._transport_kwargs: Dict[str, Any] = kwargs

    def fit(
        self,
        geom: Union[jnp.array, Geometry],
        a: Optional[jnp.array] = None,
        b: Optional[jnp.array] = None,
        **kwargs: Any,
    ) -> "UnbalancedOT":
        # TODO(michalk8): kwargs
        geom = self._prepare_geom(geom, **kwargs)
        self._transport = Transport(geom, a=a, b=b, **self._transport_kwargs)

        return self


class BaseGromowWassersteinOT(RegularizedOT, ABC):
    def __init__(
        self, cost_fn: Optional[GWLoss] = None, epsilon: Optional[Union[float, Epsilon]] = None, **kwargs: Any
    ):
        if cost_fn is None:
            cost_fn = GWSeuqCostFn()
        super().__init__(cost_fn=cost_fn, epsilon=epsilon)
        # TODO(michalk8): check args
        self._sink_kwargs: Dict[str, Any] = kwargs

    def _prepare_geom(self, geom: Union[jnp.ndarray, Geometry], **kwargs: Any) -> Geometry:
        if isinstance(geom, np.ndarray):
            geom = jnp.asarray(geom)
        if isinstance(geom, jnp.ndarray):
            # TODO(michalk8): self._cost_fn is GWLoss, not compatible with PointCloud
            geom = PointCloud(geom, cost_fn=None, epsilon=self.epsilon._init, **kwargs)
        if not isinstance(geom, Geometry):
            raise TypeError()

        return geom


class GromowWassersteinOT(GeomMixin, BaseGromowWassersteinOT):
    def fit(
        self,
        geom_a: Union[jnp.array, Geometry],
        geom_b: Union[jnp.array, Geometry],
        a: Optional[jnp.array] = None,
        b: Optional[jnp.array] = None,
        **kwargs: Any,
    ) -> "GromowWassersteinOT":
        # TODO(michalk8): handle kwargs
        geom_a = self._prepare_geom(geom_a, **kwargs)
        geom_b = self._prepare_geom(geom_b, **kwargs)

        res = gromov_wasserstein(geom_a, geom_b, a=a, b=b, loss=self._cost_fn, sinkhorn_kwargs=self._sink_kwargs)
        # TODO(michalk8): is this correct? ideally, res would contain the GW geometry object
        self._geom = Geometry(cost_matrix=res.cost_matrix, epsilon=self.epsilon._init)
        self._f = res.f
        self._g = res.g

        return self


class FusedGromowWassersteinOT(TransportMixin, BaseGromowWassersteinOT):
    def __init__(self, alpha: float = 0.5, n_iter: int = 1000, **kwargs):
        if not (0 < alpha < 1):
            raise ValueError("TODO.")

        super().__init__(**kwargs)
        self._transport: Optional[Transport] = None
        self._alpha = alpha
        self._n_iter = n_iter

    def fit(
        self,
        geom_a: Union[jnp.array, Geometry],
        geom_b: Union[jnp.array, Geometry],
        geom_ab: Union[jnp.array, Geometry],
        a: Optional[jnp.array] = None,
        b: Optional[jnp.array] = None,
        tol: float = 1e-6,
        log: bool = False,
        **kwargs: Any,
    ) -> "FusedGromowWassersteinOT":
        # TODO(michalk8): handle kwargs
        # TODO(michalk8): make sure geoms have correct metrics
        geom_a = self._prepare_geom(geom_a, **kwargs)
        geom_b = self._prepare_geom(geom_b, **kwargs)

        if a is None:
            a = jnp.ones((geom_a.shape[0],)) / geom_a.shape[0]
        if b is None:
            b = jnp.ones((geom_b.shape[0],)) / geom_b.shape[0]
        T = jnp.outer(a, b)

        # TODO(michalk8): jax.lax.scan, similar in GW
        for i in range(self._n_iter):
            G, T_prev = self._grad(self.alpha, geom_ab, geom_a, geom_b, T), T
            geom = Geometry(cost_matrix=G)
            self._transport = Transport(geom, a=a, b=b, **self._sink_kwargs)
            # TODO(michalk8): linesearch
            T = (1 - self.alpha) * T_prev + self.alpha * self._transport.matrix

            # TODO(michalk8): every n-th iter
            err = jnp.linalg.norm(T - T_prev)
            if log:
                print(i, err)

            if err <= tol:
                break

        return self

    def _grad(self, alpha: float, geom_ab: Geometry, geom_a: Geometry, geom_b: Geometry, T: jnp.ndarray) -> jnp.ndarray:
        h1 = self._cost_fn.left_x
        h2 = self._cost_fn.right_y
        C_ab = geom_ab.cost_matrix
        C_a, C_b = geom_a.cost_matrix, geom_b.cost_matrix
        # TODO(michalk8): use jax's geom
        # TODO(michalk8, Marius1311): correct sign -/+?
        return (1 - alpha) * C_ab + 2 * alpha * np.dot(h1(C_a), T).dot(h2(C_b).T)

    @property
    def alpha(self) -> float:
        """TODO."""
        return self._alpha
