import abc
import copy
import functools
from typing import Any, Callable, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

from moscot._logging import logger
from moscot._types import ArrayLike, Device_t, DTypeLike  # type: ignore[attr-defined]

__all__ = ["BaseSolverOutput", "MatrixSolverOutput"]


class BaseSolverOutput(abc.ABC):
    """Base class for all solver outputs."""

    @abc.abstractmethod
    def _apply(self, x: ArrayLike, *, forward: bool) -> ArrayLike:
        """Apply :attr:`transport_matrix` to an array of shape ``[n, d]`` or ``[m, d]``."""

    @property
    @abc.abstractmethod
    def transport_matrix(self) -> ArrayLike:
        """Transport matrix of shape ``[n, m]``."""

    @property
    @abc.abstractmethod
    def shape(self) -> Tuple[int, int]:
        """Shape of the :attr:`transport_matrix`."""

    @property
    @abc.abstractmethod
    def cost(self) -> float:
        """Regularized :term:`OT` cost."""

    @property
    @abc.abstractmethod
    def converged(self) -> bool:
        """Whether the solver converged."""

    @property
    @abc.abstractmethod
    def potentials(self) -> Optional[Tuple[ArrayLike, ArrayLike]]:
        """:term:`Dual potentials` :math:`f` and :math:`g`.

        Only valid for the :term:`Sinkhorn` algorithm.
        """

    @property
    @abc.abstractmethod
    def is_linear(self) -> bool:
        """Whether the output is a solution to a :term:`linear problem`."""

    @abc.abstractmethod
    def to(self, device: Optional[Device_t] = None) -> "BaseSolverOutput":
        """Transfer self to another compute device.

        Parameters
        ----------
        device
            Device where to transfer the solver output. If :obj:`None`, use the default device.

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
        """Whether the :attr:`transport_matrix` is :term:`low-rank`."""
        return self.rank > -1

    @abc.abstractmethod
    def _ones(self, n: int) -> ArrayLike:
        """Generate vector of 1s of shape ``[n,]``."""

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
        Array of shape ``[m,]`` or ``[m, d]``, depending on the shape of ``x``.
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
        Array of shape ``[n,]`` or ``[n, d]``, depending on the shape of ``x``.
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
            Whether to scale by :term:`marginals`.

        Returns
        -------
        The :attr:`transport_matrix` as a linear operator.
        """
        push = functools.partial(self.push, scale_by_marginals=scale_by_marginals)
        pull = functools.partial(self.pull, scale_by_marginals=scale_by_marginals)
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
            Whether to scale by :term:`marginals`.

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

        This function sets all entries of the transport matrix below a certain threshold to :math:`0` and
        returns a :class:`~moscot.base.output.MatrixSolverOutput` with sparsified transport matrix stored
        as a :class:`~scipy.sparse.csr_matrix`.

        .. warning::
            This function only serves for interfacing software which has to instantiate the transport matrix,
            :mod:`moscot` never uses the sparsified transport matrix.

        Parameters
        ----------
        mode
            How to determine the value below which entries are set to :math:`0`. Valid options are:

            - `'threshold'` - ``value`` is the threshold below which entries are set to :math:`0`.
            - `'percentile'` - ``value`` is the percentile in :math:`[0, 100]` of the :attr:`transport_matrix`.
              below which entries are set to :math:`0`.
            - `'min_row'` - ``value`` is not used, it is chosen such that each row has at least 1 non-zero entry.
        value
            Value to use for sparsification.
        batch_size
            How many rows to materialize when sparsifying the :attr:`transport_matrix`.
        n_samples
            If ``mode = 'percentile'``, determine the number of samples based on which the percentile is computed
            stochastically. Note this means that a matrix of shape `[n_samples, min(transport_matrix.shape)]`
            has to be instantiated. If `None`, ``n_samples`` is set to ``batch_size``.
        seed
            Random seed needed for sampling if ``mode = 'percentile'``.

        Returns
        -------
        Output with sparsified transport matrix.
        """
        n, m = self.shape
        if mode == "threshold":
            if value is None:
                raise ValueError("If `mode = 'threshold'`, `threshold` cannot be `None`.")
            thr = value
        elif mode == "percentile":
            if value is None:
                raise ValueError("If `mode = 'percentile'`, `threshold` cannot be `None`.")
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
                thr = min(thr, float(res.max(axis=1).min()))
        else:
            raise NotImplementedError(f"Mode `{mode}` is not yet implemented.")

        k, func, fn_stack = (n, self.push, sp.vstack) if n < m else (m, self.pull, sp.hstack)
        tmaps_sparse: List[sp.csr_matrix] = []

        for batch in range(0, k, batch_size):
            x = np.eye(k, min(batch_size, k - batch), -(min(batch, k)), dtype=float)
            res = np.array(func(x, scale_by_marginals=False))
            res[res < thr] = 0.0
            tmaps_sparse.append(sp.csr_matrix(res.T if n < m else res))

        return MatrixSolverOutput(
            transport_matrix=fn_stack(tmaps_sparse), cost=self.cost, converged=self.converged, is_linear=self.is_linear
        )

    @property
    def a(self) -> ArrayLike:
        """:term:`Marginals` of the source distribution.

        If the output of an :term:`unbalanced OT problem`, these are the posterior marginals.
        """
        return self.pull(self._ones(self.shape[1]))

    @property
    def b(self) -> ArrayLike:
        """:term:`Marginals` of the target distribution.

        If the output of an :term:`unbalanced OT problem`, these are the posterior marginals.
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
    """:term:`OT` solution with a materialized transport matrix.

    Parameters
    ----------
    transport_matrix
        Transport matrix of shape ``[n, m]``.
    cost
        Cost of an :term:`OT` problem.
    converged
        Whether the solution converged.
    is_linear
        Whether this is a solution to a :term:`linear problem`.
    """

    # TODO(michalk8): don't provide defaults?
    def __init__(
        self,
        transport_matrix: Union[ArrayLike, sp.spmatrix],
        *,
        cost: float = np.nan,
        converged: bool = True,
        is_linear: bool = True,
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

        obj = copy.copy(self)
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
