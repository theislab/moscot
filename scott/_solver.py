from abc import ABC
from typing import Any, Dict, Union, Optional

from jax import numpy as jnp
from ott.tools.transport import Transport
from ott.geometry.geometry import Geometry
from ott.geometry.pointcloud import PointCloud
from ott.core.gromov_wasserstein import GWLoss, _init_geometry_gw, gromov_wasserstein, _update_geometry_gw
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
    def __init__(self, alpha: float = 0.5, n_iter: int = 20, **kwargs):
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

        geom_ab = Geometry(cost_matrix=(1 - self.alpha) * geom_ab.cost_matrix)

        geom = _init_geometry_gw(geom_a, geom_b, a, b, epsilon=self.epsilon, loss=self._cost_fn)
        geom = Geometry(cost_matrix=geom_ab.cost_matrix + self.alpha * geom.cost_matrix)
        T, T_prev = jnp.outer(a, b), None

        # TODO(michalk8): jax.lax.scan, similar in GW
        for i in range(self._n_iter):
            self._transport = Transport(geom, a=a, b=b, **self._sink_kwargs)
            T, T_prev = self.matrix, T
            geom = self._update(geom, geom_a, geom_b, geom_ab, f=self._transport._f, g=self._transport._g)

            # TODO(michalk8): linesearch
            tau = 0.5
            T = (1 - tau) * T_prev + tau * T
            err = jnp.linalg.norm(T - T_prev)

            if log:
                print(i, err)

        self._matrix = T

        return self

    def _update(
        self, geom: Geometry, geom_a: Geometry, geom_b: Geometry, geom_ab: Geometry, f: jnp.ndarray, g: jnp.ndarray
    ) -> Geometry:
        geom_gw = _update_geometry_gw(geom, geom_a, geom_b, f=f, g=g, loss=self._cost_fn)
        tmp = geom_ab.cost_matrix + self.alpha * geom_gw.cost_matrix
        return Geometry(cost_matrix=tmp + jnp.min(tmp))

    def _linesearch(self):
        pass

    @property
    def alpha(self) -> float:
        """TODO."""
        return self._alpha
