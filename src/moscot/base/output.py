from abc import ABC, abstractmethod
from copy import copy
from functools import partial
from typing import Any, Callable, Iterable, List, Literal, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

from moscot._docs._docs import d
from moscot._logging import logger
from moscot._types import ArrayLike, Device_t, DTypeLike  # type: ignore[attr-defined]

__all__ = ["BaseSolverOutput", "MatrixSolverOutput"]


@d.dedent
class BaseSolverOutput(ABC):
    """Base class for all solver outputs."""

    @abstractmethod
    def _apply(self, x: ArrayLike, *, forward: bool) -> ArrayLike:
        pass

    @property
    @abstractmethod
    def transport_matrix(self) -> ArrayLike:
        """Transport matrix of shape ``[n, m]``."""

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int]:
        """Shape of the :attr:`transport_matrix`."""

    @property
    @abstractmethod
    def cost(self) -> float:
        """Regularized optimal transport cost."""

    @property
    @abstractmethod
    def converged(self) -> bool:
        """Whether the algorithm converged."""

    @property
    @abstractmethod
    def potentials(self) -> Optional[Tuple[ArrayLike, ArrayLike]]:
        """Dual potentials :math:`f` and :math:`g`.

        Only valid for the Sinkhorn algorithm.
        """

    @property
    @abstractmethod
    def is_linear(self) -> bool:
        """Whether the output is linear."""

    @abstractmethod
    def to(self, device: Optional[Device_t] = None) -> "BaseSolverOutput":
        """Transfer self to another device using :func:`jax.device_put`.

        Parameters
        ----------
        device
            Device where to transfer the solver output. If `None`, use the default device.

        Returns
        -------
        Self transferred to the ``device``.
        """

    @property
    def rank(self) -> int:
        """Rank of the :attr:`transport_matrix`."""
        return -1

    @property
    def is_low_rank(self) -> bool:
        """Whether the :attr:`transport_matrix` is low-rank."""
        return self.rank > -1

    # TODO(michalk8): mention in docs it needs to be broadcastable
    @abstractmethod
    def _ones(self, n: int) -> ArrayLike:
        pass

    def push(self, x: ArrayLike, scale_by_marginals: bool = False) -> ArrayLike:
        """Push mass through the :attr:`transport_matrix`.

        It is equivalent to :math:`T^T x` but without instantiating the transport matrix :math:`T`, if possible.

        Parameters
        ----------
        x
            Array of shape ``[n,]`` or ``[n, d]`` to push.
        scale_by_marginals
            Whether to scale by the source marginals :attr:`a`.

        Returns
        -------
        Array of shape ``[m,]`` or ``[m, d]``, depending on ``x``.
        """
        if x.ndim not in (1, 2):
            raise ValueError(f"Expected 1D or 2D array, found `{x.ndim}`.")
        if x.shape[0] != self.shape[0]:
            raise ValueError(f"Expected array to have shape `({self.shape[0]}, ...)`, found `{x.shape}`.")
        if scale_by_marginals:
            x = self._scale_by_marginals(x, forward=True)
        return self._apply(x, forward=True)

    def pull(self, x: ArrayLike, scale_by_marginals: bool = False) -> ArrayLike:
        """Pull mass through the :attr:`transport_matrix`.

        It is equivalent to :math:`T x` but without instantiating the transport matrix :math:`T`, if possible.

        Parameters
        ----------
        x
            Array of shape ``[m,]`` or ``[m, d]`` to pull.
        scale_by_marginals
            Whether to scale by the target marginals :attr:`b`.

        Returns
        -------
        Array of shape ``[n,]`` or ``[n, d]``, depending on ``x``.
        """
        if x.ndim not in (1, 2):
            raise ValueError(f"Expected 1D or 2D array, found `{x.ndim}`.")
        if x.shape[0] != self.shape[1]:
            raise ValueError(f"Expected array to have shape `({self.shape[1]}, ...)`, found `{x.shape}`.")
        if scale_by_marginals:
            x = self._scale_by_marginals(x, forward=False)
        return self._apply(x, forward=False)

    def as_linear_operator(self, scale_by_marginals: bool = False) -> LinearOperator:
        """Transform :attr:`transport_matrix` into a linear operator.

        Parameters
        ----------
        scale_by_marginals
            Whether to scale by marginals.

        Returns
        -------
        The :attr:`transport_matrix` as a linear operator.
        """
        push = partial(self.push, scale_by_marginals=scale_by_marginals)
        pull = partial(self.pull, scale_by_marginals=scale_by_marginals)
        # push: a @ X (rmatvec)
        # pull: X @ a (matvec)
        return LinearOperator(shape=self.shape, dtype=self.dtype, matvec=pull, rmatvec=push)

    def chain(self, outputs: Iterable["BaseSolverOutput"], scale_by_marginals: bool = False) -> LinearOperator:
        """Chain subsequent applications of :attr:`transport_matrix`.

        Parameters
        ----------
        outputs
            Sequence of transport matrices to chain.
        scale_by_marginals
            Whether to scale by marginals.

        Returns
        -------
        The chained transport matrices as a linear operator.
        """
        op = self.as_linear_operator(scale_by_marginals)
        for out in outputs:
            op *= out.as_linear_operator(scale_by_marginals)

        return op

    def sparsify(
        self,
        mode: Literal["threshold", "percentile", "min_row"],
        value: Optional[float] = None,
        batch_size: int = 1024,
        n_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> "MatrixSolverOutput":
        """Sparsify the :attr:`transport_matrix`.

        This function sets all entries of the transport matrix below `threshold` to 0 and
        returns an instance of the `MatrixSolverOutput` with the
        sparsified transport matrix stored as a :class:`~scipy.sparse.csr_matrix`.

        .. warning::
            This function only serves for interfacing software which has to instantiate the transport matrix. Methods in
            :mod:`moscot` never use the sparsified transport matrix.

        Parameters
        ----------
        mode
            How to determine the value below which entries are set to 0:

                - 'threshold' - threshold below which entries are set to 0.
                - 'percentile' - determine threshold by percentile below which entries are set to 0. Hence, between 0
                  and 100.
                - 'min_row' - choose the threshold such that each row has at least one non-zero entry.

        value
            Value to use for sparsification depending on ``mode``:

                - `'threshold'` - `value` sets the threshold below which entries are set to 0.
                - `percentile` - `value` is the percentile below which entries are set to 0.
                - `min_row` - `value` is not used.
        batch_size
            How many rows of the transport matrix to sparsify per batch.
        n_samples
            If ``mode = 'percentile'``, determine the number of samples based on which the percentile
            is computed stochastically. Note this means that a matrix of shape
            `[n_samples, min(transport_matrix.shape)]` has to be instantiated. If `None`, ``n_samples`` is set to
            ``batch_size``.
        seed
            Random seed needed for sampling if ``mode = 'percentile'``.

        Returns
        -------
        Solve output with a sparsified transport matrix.
        """
        n, m = self.shape
        if mode == "threshold":
            if value is None:
                raise ValueError("If `mode` is `threshold`, `threshold` must not be `None`.")
            thr = value
        elif mode == "percentile":
            if value is None:
                raise ValueError("If `mode` is `percentile`, `threshold` must not be `None`.")
            rng = np.random.RandomState(seed=seed)
            n_samples = n_samples if n_samples is not None else batch_size
            k = min(n_samples, n)
            x = np.zeros((m, k))
            rows = rng.choice(m, size=k)
            x[rows, np.arange(k)] = 1.0
            res = self.pull(x, scale_by_marginals=False)  # tmap @ indicator_vectors
            thr = np.percentile(res, value)
        elif mode == "min_row":
            thr = np.inf
            for batch in range(0, m, batch_size):
                x = np.eye(m, min(batch_size, m - batch), -(min(batch, m)))
                res = self.pull(x, scale_by_marginals=False)  # tmap @ indicator_vectors
                thr = min(thr, res.max(axis=1).min())
        else:
            raise NotImplementedError(mode)

        k, func, fn_stack = (n, self.push, sp.vstack) if n < m else (m, self.pull, sp.hstack)
        tmaps_sparse: List[sp.csr_matrix] = []
        for batch in range(0, k, batch_size):
            x = np.eye(k, min(batch_size, k - batch), -(min(batch, k)))
            res = func(x, scale_by_marginals=False)
            res[res < thr] = 0
            tmaps_sparse.append(sp.csr_matrix(res.T if n < m else res))
        return MatrixSolverOutput(
            transport_matrix=fn_stack(tmaps_sparse), cost=self.cost, converged=self.converged, is_linear=self.is_linear
        )

    @property
    def a(self) -> ArrayLike:
        """Marginals of the source distribution.

        If the output of an unbalanced OT problem, these are the posterior marginals.
        """
        return self.pull(self._ones(self.shape[1]))

    @property
    def b(self) -> ArrayLike:
        """Marginals of the target distribution.

        If the output of an unbalanced OT problem, these are the posterior marginals.
        """
        return self.push(self._ones(self.shape[0]))

    @property
    def dtype(self) -> DTypeLike:
        """Underlying data type."""
        return self.a.dtype

    def _scale_by_marginals(self, x: ArrayLike, *, forward: bool, eps: float = 1e-12) -> ArrayLike:
        # alt. we could use the public push/pull
        marginals = self.a if forward else self.b
        if x.ndim == 2:
            marginals = marginals[:, None]
        return x / (marginals + eps)

    def _format_params(self, fmt: Callable[[Any], str]) -> str:
        params = {"shape": self.shape, "cost": round(self.cost, 4), "converged": self.converged}
        return ", ".join(f"{name}={fmt(val)}" for name, val in params.items())

    def __bool__(self) -> bool:
        return self.converged

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self._format_params(repr)}]"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}[{self._format_params(str)}]"


class MatrixSolverOutput(BaseSolverOutput):
    """Optimal transport output with materialized `transport_matrix`.

    Parameters
    ----------
    transport_matrix
        Transport matrix of shape ``[n, m]``.
    cost
        TODO
    converged
        TODO.
    is_linear
        TODO.
    """

    # TODO(michalk8): don't provide defaults?
    def __init__(
        self, transport_matrix: ArrayLike, *, cost: float = np.nan, converged: bool = True, is_linear: bool = True
    ):
        super().__init__()
        self._transport_matrix = transport_matrix
        self._cost = cost
        self._converged = converged
        self._is_linear = is_linear

    def _apply(self, x: ArrayLike, *, forward: bool) -> ArrayLike:
        if forward:
            return self.transport_matrix.T @ x
        return self.transport_matrix @ x

    @property
    def transport_matrix(self) -> ArrayLike:  # noqa: D102
        return self._transport_matrix

    @property
    def shape(self) -> Tuple[int, int]:  # noqa: D102
        return self.transport_matrix.shape  # type: ignore[return-value]

    def to(  # noqa: D102
        self, device: Optional[Device_t] = None, dtype: Optional[DTypeLike] = None
    ) -> "BaseSolverOutput":
        if device is not None:
            logger.warning(f"`{self!r}` does not support the `device` argument, ignoring.")
        if dtype is None:
            return self

        obj = copy(self)
        obj._transport_matrix = obj.transport_matrix.astype(dtype)
        return obj

    @property
    def cost(self) -> float:  # noqa: D102
        return self._cost

    @property
    def converged(self) -> bool:  # noqa: D102
        return self._converged

    @property
    def potentials(self) -> Optional[Tuple[ArrayLike, ArrayLike]]:  # noqa: D102
        return None

    @property
    def is_linear(self) -> bool:  # noqa: D102
        return self._is_linear

    def _ones(self, n: int) -> ArrayLike:
        if isinstance(self.transport_matrix, np.ndarray):
            return np.ones((n,), dtype=self.transport_matrix.dtype)

        import jax.numpy as jnp

        return jnp.ones((n,), dtype=self.transport_matrix.dtype)
