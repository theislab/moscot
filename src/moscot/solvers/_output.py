from abc import ABC, abstractmethod
from copy import copy
from typing import Any, List, Tuple, Literal, Callable, Iterable, Optional
from functools import partial

from scipy.sparse import hstack as sparse_hstack, csr_matrix
from scipy.sparse.linalg import LinearOperator

import numpy as np

from moscot._types import Device_t, ArrayLike, DTypeLike
from moscot._logging import logger
from moscot._docs._docs import d
from moscot._constants._constants import SparsifyMode

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

        Only valid for the Sinkhorn's algorithm.
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
            Device where to transfer the solver output.
            If `None`, use the default device.

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

    @d.dedent
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

    @d.dedent
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

    @d.dedent
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

    @d.dedent
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

    @d.dedent
    def compute_sparsification(
        self,
        mode: Literal["threshold", "percentile", "min_1"] = SparsifyMode.THRESHOLD,
        threshold: float = 1e-8,
        batch_size: int = 1024,
        n_samples: Optional[int] = None,
    ) -> None:
        """
        Sparsify the transport matrix.

        This function sets all entries of the transport matrix below `threshold` to 0 and
        returns the result as a :class:`scipy.sparse.csr_matrix`.

        Parameters
        ----------
        mode
            Which threshold to use for sparsification. Can be one of:

                - "threshold" - threshold below which entries are set to 0.0.
                - "percentile" - determine threshold by percentile below which entries are set to 0. Hence, between 0
                  and 100.
                - "min_1" - choose the threshold such that each row has at least one non-zero entry.

        threshold
            Threshold or percentile depending on `mode`. If `mode` is `min_1`, `threshold` can be neglected.
        batch_size
            How many rows of the transport matrix to sparsify per batch.
        n_samples
            If `mode` is `percentile`, determine the number of samples based on which the percentile
            is computed stochastically. Note this means that a matrix of shape `[n_samples, transport_matrix.shape[1]]`
            has to be instantiated. If `None`, `n_samples` is set to batch_size.

        Returns
        -------
        Nothing, but adds the sparsified transport matrix (:class:`scipy.sparse.csr_matrix`) to `self.sparsified_tmap`.

        Note
        ----
        This function only serves for interfacing software which has to instantiate the transport matrix. Within moscot,
        there is no point in using this function. We encourage users not to use this function unless it is necessary.
        """
        mode = SparsifyMode(mode)
        if mode == SparsifyMode.THRESHOLD:
            thr = threshold
        elif mode == SparsifyMode.PERCENTILE:
            n_samples = n_samples if n_samples is not None else batch_size
            x = np.eye(self.shape[1], max(n_samples, self.shape[1]))
            res = self.pull(x, scale_by_marginals=False)  # tmap @ indicator_vectors
            thr = np.percentile(res, threshold)
        elif mode == SparsifyMode.MIN_1:
            thr = np.inf
            for batch in range(0, self.shape[1], batch_size):
                x = np.eye(self.shape[1], min(batch_size, self.shape[1] - batch), -(min(batch, self.shape[1])))
                res = self.pull(x, scale_by_marginals=False)  # tmap @ indicator_vectors
                thr_batch = res.max(axis=1).min()
                thr = thr_batch if thr_batch < thr else thr
        else:
            raise NotImplementedError(mode)

        tmaps_sparse: List[csr_matrix] = []
        for batch in range(0, self.shape[1], batch_size):
            x = np.eye(self.shape[1], min(batch_size, self.shape[1] - batch), -(min(batch, self.shape[1])))
            res = self.pull(x, scale_by_marginals=False)  # tmap @ indicator_vectors
            res[res <= thr] = 0
            tmaps_sparse.append(csr_matrix(res))
        self._sparsified_tmap = sparse_hstack(tmaps_sparse)

    @property
    def sparsified_tmap(self) -> Optional[csr_matrix]:
        return self._sparsified_tmap  # type:ignore[has-type]

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


class MatrixSolverOutput(BaseSolverOutput, ABC):
    """Optimal transport output with materialized :attr:`transport_matrix`.

    Parameters
    ----------
    transport_matrix
        Transport matrix of shape ``[n, m]``.
    """

    def __init__(self, transport_matrix: ArrayLike):
        super().__init__()
        self._transport_matrix = transport_matrix

    def _apply(self, x: ArrayLike, *, forward: bool) -> ArrayLike:
        if forward:
            return self.transport_matrix.T @ x
        return self.transport_matrix @ x

    @property
    def transport_matrix(self) -> ArrayLike:
        return self._transport_matrix

    @property
    def shape(self) -> Tuple[int, int]:
        return self.transport_matrix.shape  # type: ignore[return-value]

    def to(self, device: Optional[Device_t] = None, dtype: Optional[DTypeLike] = None) -> "BaseSolverOutput":
        if device is not None:
            logger.warning(f"`{self!r}` does not support the `device` argument, ignoring.")
        if dtype is None:
            return self

        obj = copy(self)
        obj._transport_matrix = obj.transport_matrix.astype(dtype)
        return obj
