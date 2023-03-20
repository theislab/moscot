from abc import ABC, abstractmethod
from copy import copy
from functools import partial
from typing import Any, Callable, Iterable, Optional, Tuple

import numpy as np
from scipy.sparse.linalg import LinearOperator

from moscot._logging import logger
from moscot._types import ArrayLike, Device_t, DTypeLike  # type: ignore[attr-defined]

__all__ = ["BaseSolverOutput", "MatrixSolverOutput"]


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
        """Transfer self to another device.

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
    """Optimal transport output with materialized transport matrix.

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