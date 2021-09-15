from abc import ABC
from typing import Any, Dict, Union, Optional

from jax import numpy as jnp
from ott.geometry.costs import CostFn, Euclidean
from ott.tools.transport import Transport
from ott.geometry.geometry import Geometry
from ott.geometry.pointcloud import PointCloud
from ott.core.gromov_wasserstein import GWLoss, GWSqEuclLoss, gromov_wasserstein, _marginal_dependent_cost
from ott.geometry.epsilon_scheduler import Epsilon
import numpy as np

from moscot._base import CostFn_t, BaseSolver
from moscot._mixins import SimpleMixin, TransportMixin


class RegularizedOT(BaseSolver, ABC):
    def __init__(self, cost_fn: Optional[CostFn_t] = None, epsilon: Optional[Union[float, Epsilon]] = None):
        super().__init__(cost_fn=cost_fn)
        self._epsilon: Epsilon = epsilon if isinstance(epsilon, Epsilon) else Epsilon(target=epsilon)

    # TODO(michalk8): refactor this
    @property
    def _default_cost_fn(self) -> Union[CostFn, GWLoss]:
        return Euclidean()

    @property
    def epsilon(self) -> Epsilon:
        """Regularization parameter."""
        return self._epsilon

    # TODO(michalk8): if array, allow it to be kernel/distance matrix
    # i.e. distance=True, kernel=True, only 1 allowed to be True
    def _prepare_geom(self, geom: Union[jnp.ndarray, Geometry], **kwargs: Any) -> Geometry:
        if isinstance(geom, np.ndarray):
            geom = jnp.asarray(geom)
        if isinstance(geom, jnp.ndarray):
            geom = PointCloud(geom, cost_fn=self._cost_fn, epsilon=self.epsilon, **kwargs)
        if not isinstance(geom, Geometry):
            raise TypeError()

        return geom


# TODO(michalk8): find a more suitable name (balanced case when tau_a=1.0, tau_b=1.0)
class Unbalanced(TransportMixin, RegularizedOT):
    """
    Unbalanced entropy-regularized OT.

    Parameters
    ----------
    cost_fn
        Cost function to use. Default is euclidean.
    epsilon
        Regularization parameter.
    kwargs
        Keyword arguments for :func:`ott.core.sinkhorn.sinkhorn`.
    """

    def __init__(
        self, cost_fn: Optional[CostFn_t] = None, epsilon: Optional[Union[float, Epsilon]] = None, **kwargs: Any
    ):
        super().__init__(cost_fn=cost_fn, epsilon=epsilon)
        self._transport_kwargs: Dict[str, Any] = kwargs

    def fit(
        self,
        geom: Union[jnp.array, Geometry],
        a: Optional[jnp.array] = None,
        b: Optional[jnp.array] = None,
        **kwargs: Any,
    ) -> "Unbalanced":
        """
        Computer unbalanced entropy-regularized OT.

        Parameters
        ----------
        geom
            Geometry.
        a
            Source weights of ``geom``.
        b
            Target weights of ``geom``.
        kwargs
            Keyword arguments for :class:`ott.geometry.pointcloud.PointCloud`.
            Only used if any geometries are specified as :class:`numpy.ndarray` or :class:`jax.numpy.ndarray`.

        Returns
        -------
        Fitted self.
        """
        geom = self._prepare_geom(geom, **kwargs)
        self._transport = Transport(geom, a=a, b=b, **self._transport_kwargs)

        return self


class BaseGW(RegularizedOT, ABC):
    def __init__(
        self, cost_fn: Optional[GWLoss] = None, epsilon: Optional[Union[float, Epsilon]] = None, **kwargs: Any
    ):
        super().__init__(cost_fn=cost_fn, epsilon=epsilon)
        if not isinstance(self._cost_fn, GWLoss):
            raise TypeError(
                f"Expected the cost function to be of type `{GWLoss.__name__}`, found `{type(self._cost_fn)}`."
            )
        self._sink_kwargs: Dict[str, Any] = kwargs

    @property
    def _default_cost_fn(self) -> Union[CostFn, GWLoss]:
        return GWSqEuclLoss()

    def _prepare_geom(self, geom: Union[jnp.ndarray, Geometry], **kwargs: Any) -> Geometry:
        if isinstance(geom, np.ndarray):
            geom = jnp.asarray(geom)
        if isinstance(geom, jnp.ndarray):
            cost_fn = Euclidean if isinstance(self._cost_fn, GWSqEuclLoss) else None
            # TODO(michalk8): this will always be euclidean (if passing None), nicer solution
            geom = PointCloud(geom, cost_fn=cost_fn, epsilon=self.epsilon, **kwargs)
        if not isinstance(geom, Geometry):
            raise TypeError()

        return geom


class GW(SimpleMixin, BaseGW):
    """
    Gromov-Wasserstein OT.

    Parameters
    ----------
    cost_fn
        Cost function to use. Default is euclidean.
    epsilon
        Regularization parameter.
    kwargs
        Keyword arguments for :func:`ott.core.sinkhorn.sinkhorn`.

    References
    ----------
    :cite:`memoli:2011` :cite:`peyre:2016`
    """

    def __init__(
        self, cost_fn: Optional[GWLoss] = None, epsilon: Optional[Union[float, Epsilon]] = None, **kwargs: Any
    ):
        super().__init__(cost_fn=cost_fn, epsilon=epsilon, **kwargs)

    def fit(
        self,
        geom_a: Union[jnp.array, Geometry],
        geom_b: Union[jnp.array, Geometry],
        a: Optional[jnp.array] = None,
        b: Optional[jnp.array] = None,
        **kwargs: Any,
    ) -> "GW":
        """
        Compute GW.

        Parameters
        ----------
        geom_a
            First geometry.
        geom_b
            Second geometry.
        a
            Weights of ``geom_a``.
        b
            Weights of ``geom_b``.
        kwargs
            Keyword arguments for :class:`ott.geometry.pointcloud.PointCloud`.
            Only used if any geometries are specified as :class:`numpy.ndarray` or :class:`jax.numpy.ndarray`.

        Returns
        -------
        Fitted self.
        """
        geom_a = self._prepare_geom(geom_a, **kwargs)
        geom_b = self._prepare_geom(geom_b, **kwargs)

        res = gromov_wasserstein(geom_a, geom_b, a=a, b=b, loss=self._cost_fn, sinkhorn_kwargs=self._sink_kwargs)
        self._matrix = res.transport

        return self


class FusedGW(SimpleMixin, BaseGW):
    """
    Fused Gromov-Wasserstein OT.

    Parameters
    ----------
    alpha
        Weight of GW. Must be in `(0, 1)`.
    cost_fn
        Cost function to use. Default is euclidean.
    epsilon
        Regularization parameter.
    kwargs
        Keyword arguments for :func:`ott.core.sinkhorn.sinkhorn`.

    References
    ----------
    :cite:`vayer:2019` :cite:`nitzan:2019`
    """

    def __init__(self, alpha: float = 0.5, **kwargs: Any):
        if not (0 < alpha < 1):
            raise ValueError(f"Expected `alpha` to be in interval `(0, 1)`, found `{alpha}`.")

        super().__init__(**kwargs)
        self._alpha = alpha

    def fit(
        self,
        geom_a: Union[jnp.array, Geometry],
        geom_b: Union[jnp.array, Geometry],
        geom_ab: Union[jnp.array, Geometry],
        a: Optional[jnp.array] = None,
        b: Optional[jnp.array] = None,
        n_iters: int = 20,
        tol: float = 1e-6,
        linesearch: bool = True,
        log: bool = True,
        **kwargs: Any,
    ) -> "FusedGW":
        """
        Compute Fused Gromov-Wasserstein OT.

        Parameters
        ----------
        geom_a
            First geometry.
        geom_b
            Second geometry.
        geom_ab
            Joint geometry.
        a
            Weights of ``geom_a``.
        b
            Weights of ``geom_b``.
        n_iters
            Number of iterations.
        tol
            Tolerance stopping criterion.
        linesearch
            Whether to perform line search to find :math:`\tau` as described in :cite:`vayer:2019`.
        log
            Whether to log.
        kwargs
            Keyword arguments for :class:`ott.geometry.pointcloud.PointCloud`.
            Only used if any geometries are specified as :class:`numpy.ndarray` or :class:`jax.numpy.ndarray`.

        Returns
        -------
        Fitted self.
        """
        geom_a = self._prepare_geom(geom_a, **kwargs)
        geom_b = self._prepare_geom(geom_b, **kwargs)
        geom_ab = self._prepare_geom(geom_ab, **kwargs)

        if a is None:
            a = jnp.ones((geom_a.shape[0],)) / geom_a.shape[0]
        if b is None:
            b = jnp.ones((geom_b.shape[0],)) / geom_b.shape[0]

        C12 = self._marginal_dep_term(geom_a, geom_b, a, b)
        T, T_hat = jnp.outer(a, b), None
        geom_ab = Geometry(cost_matrix=(1 - self.alpha) * geom_ab.cost_matrix)

        # TODO(michalk8): jax.lax.scan, similar in GW in ott
        for i in range(n_iters):
            geom = self._update(geom_a, geom_b, geom_ab, T, C12=C12)
            transport = Transport(geom, a=a, b=b, **self._sink_kwargs)
            T_hat = transport.matrix

            if linesearch:
                tau = self._linesearch(geom_a, geom_b, geom_ab, T=T, T_hat=T_hat, C12=C12)
            else:
                tau = 1.0
            err = jnp.linalg.norm(T - T_hat)
            if log:
                print(f"{i + 1}. err={err} tau={tau}")

            T = (1 - tau) * T + tau * T_hat
            if err < tol:
                break

        self._matrix = T

        return self

    def _update(
        self,
        geom_a: Geometry,
        geom_b: Geometry,
        geom_ab: Geometry,
        T: jnp.ndarray,
        C12: jnp.ndarray,
    ) -> Geometry:
        h1 = self._cost_fn.left_x
        h2 = self._cost_fn.right_y
        C_ab = geom_ab.cost_matrix
        C_a, C_b = geom_a.cost_matrix, geom_b.cost_matrix
        # references:
        # https://github.com/PythonOT/POT/blob/master/ot/gromov.py#L205
        # https://github.com/PythonOT/POT/blob/master/ot/gromov.py#L137-L138
        tmp = C_ab + self.alpha * 2 * (C12 - np.dot(h1(C_a), T).dot(h2(C_b).T))
        return Geometry(cost_matrix=tmp, epsilon=self.epsilon)

    def _marginal_dep_term(self, geom_a: Geometry, geom_b: Geometry, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        # TODO(michalk8): taken from ott, we could be more mem. efficient
        ab = a[:, None] * b[None, :]
        marginal_x = ab.sum(1)
        marginal_y = ab.sum(0)
        return _marginal_dependent_cost(marginal_x, marginal_y, geom_a, geom_b, self._cost_fn)

    def _linesearch(
        self,
        geom_a: Geometry,
        geom_b: Geometry,
        geom_ab: Geometry,
        T: jnp.ndarray,
        T_hat: jnp.ndarray,
        C12: jnp.ndarray,
    ) -> float:
        # TODO(michalk8): look into armijo line search
        # see: https://hal.archives-ouvertes.fr/hal-02971153/document
        # algorithm 2 for explanation or
        # https://github.com/PythonOT/POT/blob/c105dcb892de87ae9c6cfcfc5d9c0b14f2933082/ot/optim.py#L418
        M = geom_ab.cost_matrix
        C1 = geom_a.cost_matrix
        C2 = geom_b.cost_matrix
        tmp = jnp.dot(jnp.dot(C1, T_hat), C2)

        a = -2 * self.alpha * jnp.sum(tmp * T_hat)
        b = jnp.sum((M + self.alpha * C12) * T_hat)
        b -= 2 * self.alpha * (jnp.sum(tmp * T) + jnp.sum(jnp.dot(jnp.dot(C1, T), C2) * T_hat))

        if a > 0:
            return float(min(1.0, max(0.0, -b / (2.0 * a))))

        return float(a + b < 0)

    @property
    def alpha(self) -> float:
        """Weight of Gromov-Wasserstein."""
        return self._alpha
