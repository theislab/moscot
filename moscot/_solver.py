from abc import ABC
from typing import Any, Dict, List, Union, Callable, Optional
import warnings

from typing_extensions import Literal

from jax import numpy as jnp, random
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
    def __init__(
        self, cost_fn: Optional[CostFn_t] = None, epsilon: Optional[Union[float, Epsilon]] = None, **kwargs: Any
    ):
        super().__init__(cost_fn=cost_fn)
        self._epsilon: Epsilon = epsilon if isinstance(epsilon, (Epsilon, type(None))) else Epsilon(target=epsilon)

        kwargs.setdefault("jit", True)
        self._kwargs: Dict[str, Any] = kwargs

    # TODO(michalk8): refactor this
    @property
    def _default_cost_fn(self) -> Union[CostFn, GWLoss]:
        return Euclidean()

    @property
    def epsilon(self) -> Optional[Epsilon]:
        """Regularization parameter."""
        return self._epsilon

    # TODO(michalk8): if array, allow it to be kernel/distance matrix
    # i.e. distance=True, kernel=True, only 1 allowed to be True
    def _prepare_geom(self, geom: Union[jnp.ndarray, Geometry], **kwargs: Any) -> Geometry:
        if isinstance(geom, Geometry):
            # TODO(michalk8): not efficient
            return Geometry(cost_matrix=geom.cost_matrix, epsilon=self.epsilon)

        if isinstance(geom, np.ndarray):
            geom = jnp.asarray(geom)
        if isinstance(geom, jnp.ndarray):
            geom = PointCloud(geom, cost_fn=self._cost_fn, epsilon=self.epsilon, **kwargs)
        if not isinstance(geom, Geometry):
            raise TypeError(type(geom))

        return geom


class Regularized(TransportMixin, RegularizedOT):
    """
    Entropy-regularized OT.

    Parameters
    ----------
    cost_fn
        Cost function to use. Default is euclidean.
    epsilon
        Regularization parameter.
    kwargs
        Keyword arguments for :func:`ott.core.sinkhorn.sinkhorn`.
    """

    def fit(
        self,
        geom: Union[jnp.array, Geometry],
        a: Optional[jnp.array] = None,
        b: Optional[jnp.array] = None,
        **kwargs: Any,
    ) -> "Regularized":
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
        self._transport = Transport(geom, a=a, b=b, **self._kwargs)
        self._check_marginals(a, b)

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
        self._converged_sinkhorn: List[bool] = []

    @property
    def converged_sinkhorn(self) -> List[bool]:
        """Convergence of each sinkhorn iteration."""
        return self._converged_sinkhorn

    @property
    def _default_cost_fn(self) -> Union[CostFn, GWLoss]:
        return GWSqEuclLoss()

    def _prepare_geom(self, geom: Union[jnp.ndarray, Geometry], **kwargs: Any) -> Geometry:
        if isinstance(geom, np.ndarray):
            geom = jnp.asarray(geom)
        if isinstance(geom, jnp.ndarray):
            cost_fn = Euclidean() if isinstance(self._cost_fn, GWSqEuclLoss) else None
            # TODO(michalk8): this will always be euclidean (if passing None), nicer solution
            geom = PointCloud(geom, geom, cost_fn=cost_fn, epsilon=self.epsilon, **kwargs)
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
        # TODO(michalk8): shall we keep/not keep jit for sinkhorn? does it matter?
        self._jit = kwargs.get("jit", True)
        self._max_iterations = kwargs.pop("max_iterations", 20)
        self._warm_start = kwargs.pop("warm_start", True)
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

        res = gromov_wasserstein(
            geom_a,
            geom_b,
            a=a,
            b=b,
            epsilon=self.epsilon,
            loss=self._cost_fn,
            max_iterations=self._max_iterations,
            jit=self._jit,
            warm_start=self._warm_start,
            sinkhorn_kwargs=self._kwargs,
        )
        self._matrix = res.transport
        self._converged = all(res.converged_sinkhorn)
        self._converged_sinkhorn = list(map(bool, res.converged_sinkhorn))
        self._check_marginals(a, b)

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

    # TODO(michalk8): move max_iterations to init?
    def fit(
        self,
        geom_a: Union[jnp.array, Geometry],
        geom_b: Union[jnp.array, Geometry],
        geom_ab: Union[jnp.array, Geometry],
        a: Optional[jnp.array] = None,
        b: Optional[jnp.array] = None,
        max_iterations: int = 20,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        init_method: Literal["uniform", "random"] = "uniform",
        scale_fn: Optional[Callable[[jnp.array], float]] = None,
        linesearch: bool = True,
        seed: Optional[int] = None,
        verbose: bool = True,
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
        max_iterations
            Number of iterations.
        rtol
            Relative tolerance stopping criterion.
        atol
            Relative tolerance stopping criterion.
        init_method
            How to initialize the coupling.
        scale_fn
            How to scale cost matrix terms.
        linesearch
            Whether to perform line search to find :math:`\tau` as described in :cite:`vayer:2019`.
        seed
            Random seed when ``init_method = 'random'``.
        verbose
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
        T, T_hat, f_val = self._get_initial_coupling(a, b, method=init_method, seed=seed), None, 0
        converged = []

        fmt = "{:5s}|{:12s}|{:8s}|{:8s}|{:8s}|{:8s}|{:8s}"
        if verbose:
            print(
                fmt.format(
                    "It.",
                    "Loss",
                    "Rel. loss   ",
                    "Abs. loss   ",
                    "tau         ",
                    "converged   ",
                    "eps         ",
                )
                + "\n"
                + "-" * 83
            )

        scale_ab = float(1.0 if scale_fn is None else scale_fn(geom_ab.cost_matrix))
        geom_ab = Geometry(cost_matrix=(1 - self.alpha) * geom_ab.cost_matrix / scale_ab)

        # TODO(michalk8): jax.lax.scan, similar in GW in ott
        self._converged = True
        relative_delta_fval, abs_delta_fval = np.inf, np.inf

        for i in range(max_iterations):
            old_fval = f_val
            geom = self._update(geom_a, geom_b, geom_ab, T, C12=C12, scale_fn=scale_fn)

            transport = Transport(geom, a=a, b=b, **self._kwargs)
            T_hat = transport.matrix
            converged.append(bool(transport.converged))

            # TODO(michalk8): will need to eval the cost after line search if tau != 1.0
            f_val = transport.reg_ot_cost
            abs_delta_fval = abs(f_val - old_fval)
            relative_delta_fval = abs_delta_fval / abs(f_val)

            if linesearch:
                tau = self._linesearch(geom_a, geom_b, geom_ab, T=T, T_hat=T_hat, C12=C12)
            else:
                tau = 1.0
            T = (1 - tau) * T + tau * T_hat

            if verbose:
                print(
                    f"{i + 1:5d}|{f_val:8e}|{relative_delta_fval:8e}|{abs_delta_fval:8e}|"
                    f"{tau:8e}|{transport.converged:12d}|{geom.epsilon:8e}"
                )

            if relative_delta_fval <= rtol or abs_delta_fval <= atol:
                break
        else:
            self._converged = False

        self._matrix = T
        self._converged_sinkhorn = converged

        if not self.converged:
            warnings.warn(
                f"Maximum number of iterations ({max_iterations}) reached "
                f"with `rtol={relative_delta_fval:.4e}`, `atol={abs_delta_fval:.4e}`",
                category=UserWarning,
            )
        self._check_marginals(a, b)

        return self

    def _get_initial_coupling(
        self,
        a: jnp.ndarray,
        b: jnp.ndarray,
        *,
        method: Literal["uniform", "random"],
        seed: Optional[int] = None,
    ) -> jnp.ndarray:
        seed = np.random.RandomState(seed).randint(0, 2 ** 32 - 1)
        key = random.PRNGKey(seed)

        if method == "uniform":
            return jnp.outer(a, b)
        if method == "random":
            # TODO(michalk8): RecursionError in `ott` if epsilon is `None`
            if self.epsilon is None:
                raise ValueError("Please specify `epsilon=...` when using `init_method='random'.`")
            geom = Geometry(
                kernel_matrix=random.uniform(key, shape=(len(a), len(b)), dtype=float), epsilon=self.epsilon
            )
            return Transport(geom, a=a, b=b).matrix

        raise NotImplementedError(method)

    def _update(
        self,
        geom_a: Geometry,
        geom_b: Geometry,
        geom_ab: Geometry,
        T: jnp.ndarray,
        C12: jnp.ndarray,
        scale_fn: Optional[Callable[[jnp.array], float]] = None,
    ) -> Geometry:
        h1 = self._cost_fn.left_x
        h2 = self._cost_fn.right_y
        C_ab = geom_ab.cost_matrix
        C_a, C_b = geom_a.cost_matrix, geom_b.cost_matrix
        # references:
        # https://github.com/PythonOT/POT/blob/master/ot/gromov.py#L205
        # https://github.com/PythonOT/POT/blob/master/ot/gromov.py#L137-L138
        C = 2 * (C12 - np.dot(h1(C_a), T).dot(h2(C_b).T))
        scale_c = float(1.0 if scale_fn is None else scale_fn(C))
        # tmp = C_ab + self.alpha * 2 * (C12 - np.dot(h1(C_a), T).dot(h2(C_b).T))
        return Geometry(cost_matrix=C_ab + self.alpha * (C / scale_c), epsilon=self.epsilon)

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
