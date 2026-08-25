from __future__ import annotations

import abc
import copy
import functools
from typing import Any, Callable, Iterable, Literal, Optional, Union, cast

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

from moscot._logging import logger
from moscot._types import ArrayLike, Device_t, DTypeLike

__all__ = ["BaseDiscreteSolverOutput", "MatrixSolverOutput"]


def _mass_select_block(rows: np.ndarray, *, value: float, max_k: Optional[int]) -> sp.csr_matrix:
    """Keep, per row, the smallest set of entries capturing ``value`` of the row mass.

    Parameters
    ----------
    rows
        Dense block of shape ``[b, m]``; each row is a row of the transport matrix.
    value
        Target fraction of each row's mass to retain, in ``(0, 1]``.
    max_k
        Optional cap on the number of entries kept per row.

    Returns
    -------
    Sparsified block as a :class:`~scipy.sparse.csr_matrix` of shape ``[b, m]``.
    """
    _, m = rows.shape
    order = np.argsort(rows, axis=1)[:, ::-1]  # descending
    sorted_vals = np.take_along_axis(rows, order, axis=1)
    totals = sorted_vals.sum(axis=1, keepdims=True)
    cum = np.cumsum(sorted_vals, axis=1)
    # `value >= 1` keeps everything (avoids float cumsum vs. sum mismatch -> exact reconstruction).
    target = np.full_like(totals, np.inf) if value >= 1.0 else value * totals
    # smallest prefix whose cumulative mass reaches the target (crossing entry included).
    k_per_row = (cum < target).sum(axis=1) + 1
    k_per_row = np.where(totals.ravel() <= 0.0, 0, k_per_row)  # all-zero rows keep nothing
    if max_k is not None:
        k_per_row = np.minimum(k_per_row, max_k)
    k_per_row = np.minimum(k_per_row, m)
    keep_sorted = np.arange(m)[None, :] < k_per_row[:, None]
    sel = np.zeros(rows.shape, dtype=bool)
    np.put_along_axis(sel, order, keep_sorted, axis=1)
    return sp.csr_matrix(np.where(sel, rows, 0.0))


def _massless_rows(rows: np.ndarray) -> np.ndarray:
    """Mask of rows carrying no mass.

    Such rows cannot be kept non-empty by any threshold, so every sparsification mode treats them
    the same way: they keep no entries, and they are excluded from the statistics that determine a
    threshold. Letting them take part would drag any row-based threshold down to :math:`0` and
    disable sparsification altogether, without making the row usable.
    """
    return ~(rows.max(axis=1) > 0.0)


def _sparsify_block(
    rows: np.ndarray, *, mode: str, thr: Optional[float], value: Optional[float], max_k: Optional[int]
) -> sp.csr_matrix:
    """Apply the sparsification criterion to a block of transport-matrix rows."""
    if mode == "mass":
        assert value is not None  # validated in `sparsify`
        return _mass_select_block(rows, value=value, max_k=max_k)
    rows = np.array(rows)  # writable copy
    rows[rows < thr] = 0.0
    return sp.csr_matrix(rows)


class BaseSolverOutput(abc.ABC):
    """Base class for all solver outputs."""

    @abc.abstractmethod
    def push(self, x: ArrayLike, **kwargs) -> ArrayLike:
        """Push the solution based on a condition."""

    @abc.abstractmethod
    def _apply_forward(self, x: ArrayLike) -> ArrayLike:
        """Apply the transport matrix in the forward direction."""

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, int]:
        """Shape of the problem."""

    @property
    @abc.abstractmethod
    def converged(self) -> bool:
        """Whether the solver converged."""

    @abc.abstractmethod
    def to(self: BaseSolverOutput, device: Optional[Device_t] = None) -> BaseSolverOutput:
        """Transfer self to another compute device.

        Parameters
        ----------
        device
            Device where to transfer the solver output. If :obj:`None`, use the default device.

        Returns
        -------
        Self transferred to the ``device``.
        """

    def _format_params(self, fmt: Callable[[Any], str]) -> str:
        params = {"shape": self.shape}
        return ", ".join(f"{name}={fmt(val)}" for name, val in params.items())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self._format_params(repr)}]"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}[{self._format_params(str)}]"


class BaseDiscreteSolverOutput(BaseSolverOutput, abc.ABC):
    """Base class for all discrete solver outputs."""

    @abc.abstractmethod
    def _apply(self, x: ArrayLike, *, forward: bool) -> ArrayLike:
        """Apply :attr:`transport_matrix` to an array of shape ``[n, d]`` or ``[m, d]``."""

    @property
    @abc.abstractmethod
    def transport_matrix(self) -> ArrayLike:
        """Transport matrix of shape ``[n, m]``."""

    @property
    @abc.abstractmethod
    def cost(self) -> float:
        """Regularized :term:`OT` cost."""

    @property
    @abc.abstractmethod
    def potentials(self) -> Optional[tuple[ArrayLike, ArrayLike]]:
        """:term:`Dual potentials` :math:`f` and :math:`g`.

        Only valid for the :term:`Sinkhorn` algorithm.
        """

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, int]:
        """Shape of the :attr:`transport_matrix`."""

    @property
    @abc.abstractmethod
    def is_linear(self) -> bool:
        """Whether the output is a solution to a :term:`linear problem`."""

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

    def chain(self, outputs: Iterable[BaseDiscreteSolverOutput], scale_by_marginals: bool = False) -> LinearOperator:
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

    def _materialize_rows(self, ixs: np.ndarray) -> ArrayLike:
        """Materialize rows ``ixs`` of the :attr:`transport_matrix` as a dense ``[len(ixs), m]`` array.

        This default pushes indicator columns, which is correct for any output but costs
        ``n * m * len(ixs)`` whenever :meth:`push` materializes a dense tensor - as :mod:`ott`'s
        log-sum-exp apply does. Subclasses that can slice or recompute rows directly override it,
        caching whatever per-output work they need in :attr:`_row_source`.
        """
        n = self.shape[0]
        x = np.zeros((n, len(ixs)), dtype=float)
        x[ixs, np.arange(len(ixs))] = 1.0
        return np.asarray(self.push(x, scale_by_marginals=False)).T

    def sparsify(
        self,
        mode: Literal["threshold", "percentile", "min_row", "mass"],
        value: Optional[float] = None,
        batch_size: int = 1024,
        n_samples: Optional[int] = None,
        seed: Optional[int] = None,
        max_k: Optional[int] = None,
    ) -> MatrixSolverOutput:
        """Sparsify the :attr:`transport_matrix`.

        This function sets entries of the transport matrix to :math:`0` according to ``mode`` and
        returns a :class:`~moscot.base.output.MatrixSolverOutput` with sparsified transport matrix stored
        as a :class:`~scipy.sparse.csr_matrix`. The transport matrix is materialized in row blocks of
        ``batch_size`` rows; outputs that can build rows directly - :mod:`ott` :term:`Sinkhorn`,
        :term:`Gromov-Wasserstein`/:term:`fused Gromov-Wasserstein`, their :term:`low-rank` counterparts,
        and already materialized matrices - never hold more than ``[batch_size, m]`` at a time. Other
        outputs fall back to pushing indicator columns, whose cost depends on the output's :meth:`push`.

        Rows carrying no mass keep no entries and take no part in choosing a threshold, whichever
        ``mode`` is used - no threshold can make them non-empty, and letting them take part would
        drag the threshold down to :math:`0`. A warning is emitted when any are present, since the
        result then cannot be normalized into a Markov transition matrix.

        .. warning::
            This function only serves for interfacing software which has to instantiate the transport matrix,
            :mod:`moscot` never uses the sparsified transport matrix.

        Parameters
        ----------
        mode
            How to determine the entries that are set to :math:`0`. Valid options are:

            - `'threshold'` - ``value`` is the threshold below which entries are set to :math:`0`.
            - `'percentile'` - ``value`` is the percentile in :math:`[0, 100]` of the :attr:`transport_matrix`.
              below which entries are set to :math:`0`, estimated from ``n_samples`` randomly sampled rows.
            - `'min_row'` - ``value`` is not used, it is set to ``min_i max_j T_ij``, the largest
              threshold for which every row keeps at least 1 non-zero entry. It does not depend on
              ``batch_size``; the rows are materialized twice, once for the threshold and once for the values.
            - `'mass'` - per row, keep the largest entries capturing a fraction ``value`` of the row's mass
              (at most ``max_k`` entries per row); ``value`` must be in :math:`(0, 1]`.
        value
            Value to use for sparsification. Its meaning depends on ``mode`` (see above).
        batch_size
            How many rows to materialize at a time when sparsifying the :attr:`transport_matrix`.
        n_samples
            If ``mode = 'percentile'``, the number of rows sampled to estimate the percentile stochastically.
            Note this means that a matrix of shape `[n_samples, m]` has to be instantiated.
            If `None`, ``n_samples`` is set to ``batch_size``.
        seed
            Random seed needed for sampling if ``mode = 'percentile'``.
        max_k
            Maximum number of entries to keep per row. Only valid when ``mode = 'mass'``.

        Returns
        -------
        Output with sparsified transport matrix.
        """
        n, m = self.shape
        row_blocks = [np.arange(start, min(start + batch_size, n)) for start in range(0, n, batch_size)]

        thr: Optional[float] = None
        if mode == "mass":
            if value is None or not 0.0 < value <= 1.0:
                raise ValueError("If `mode = 'mass'`, `value` must be in `(0, 1]`.")
            if max_k is not None and max_k <= 0:
                raise ValueError(f"`max_k` must be a positive integer, found `{max_k}`.")
        elif mode == "threshold":
            if value is None:
                raise ValueError("If `mode = 'threshold'`, `value` cannot be `None`.")
            thr = value
        elif mode == "percentile":
            if value is None:
                raise ValueError("If `mode = 'percentile'`, `value` cannot be `None`.")
            rng = np.random.RandomState(seed=seed)
            k = min(n_samples if n_samples is not None else batch_size, n)
            ixs = np.sort(rng.choice(n, size=k, replace=False))
            sample = np.asarray(self._materialize_rows(ixs))
            sample = sample[~_massless_rows(sample)]
            thr = float(np.percentile(sample, value)) if sample.size else np.inf
        elif mode == "min_row":
            # exact rule `min_i max_j T_ij`: every block holds whole rows, so each row's maximum is
            # exact and the threshold does not depend on `batch_size`. Computing it from the same
            # blocks that produce the values also keeps the two consistent to the last bit.
            thr = np.inf
            for ixs in row_blocks:
                maxima = np.asarray(self._materialize_rows(ixs)).max(axis=1)
                maxima = maxima[maxima > 0.0]  # see `_massless_rows`
                if maxima.size:
                    thr = min(thr, float(maxima.min()))
        else:
            raise NotImplementedError(f"Mode `{mode}` is not yet implemented.")

        if mode != "mass" and max_k is not None:
            raise ValueError("`max_k` is only supported with `mode = 'mass'`.")

        # Always iterate over source rows so that each block holds rows of the transport matrix.
        # This keeps the per-row criteria (e.g. `mass`) well-defined and peak memory at `[batch_size, m]`.
        tmaps_sparse: list[sp.csr_matrix] = []
        n_massless = 0
        for ixs in row_blocks:
            block = np.asarray(self._materialize_rows(ixs))
            n_massless += int(_massless_rows(block).sum())
            tmaps_sparse.append(_sparsify_block(block, mode=mode, thr=thr, value=value, max_k=max_k))
        if n_massless:
            logger.warning(
                f"`{n_massless}` row(s) of the transport matrix carry no mass and are left empty; "
                f"the sparsified matrix cannot be normalized into a Markov transition matrix."
            )

        transport_matrix = sp.vstack(tmaps_sparse).tocsr() if tmaps_sparse else sp.csr_matrix((n, m))
        return MatrixSolverOutput(
            transport_matrix=transport_matrix,
            cost=self.cost,
            converged=self.converged,
            is_linear=self.is_linear,
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

    def _format_params(self, fmt: Callable[[Any], str]) -> str:
        params = {"shape": self.shape, "cost": round(self.cost, 4), "converged": self.converged}
        return ", ".join(f"{name}={fmt(val)}" for name, val in params.items())

    def _scale_by_marginals(self, x: ArrayLike, *, forward: bool, eps: float = 1e-12) -> ArrayLike:
        # alt. we could use the public push/pull
        marginals = self.a if forward else self.b
        if x.ndim == 2:
            marginals = marginals[:, None]
        return x / (marginals + eps)

    def __bool__(self) -> bool:
        return self.converged


class MatrixSolverOutput(BaseDiscreteSolverOutput):
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

    def _apply_forward(self, x: ArrayLike) -> ArrayLike:
        return self._apply(x, forward=True)

    @functools.cached_property
    def _row_source(self) -> Union[sp.spmatrix, np.ndarray]:
        """The transport matrix, resolved once: sparse as-is, anything else (e.g. :mod:`jax`) as dense."""
        tmap = self.transport_matrix
        return cast(sp.spmatrix, tmap) if sp.issparse(tmap) else np.asarray(tmap)

    def _materialize_rows(self, ixs: np.ndarray) -> ArrayLike:  # noqa: D102
        rows = self._row_source[ixs]
        return rows.toarray() if sp.issparse(rows) else rows

    @property
    def transport_matrix(self) -> ArrayLike:  # noqa: D102
        return self._transport_matrix

    @property
    def shape(self) -> tuple[int, ...]:  # noqa: D102
        return self.transport_matrix.shape

    def to(  # noqa: D102
        self, device: Optional[Device_t] = None, dtype: Optional[DTypeLike] = None
    ) -> BaseDiscreteSolverOutput:
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
    def potentials(self) -> Optional[tuple[ArrayLike, ArrayLike]]:  # noqa: D102
        return None

    @property
    def is_linear(self) -> bool:  # noqa: D102
        return self._is_linear

    def _ones(self, n: int) -> ArrayLike:
        if isinstance(self.transport_matrix, np.ndarray):
            return np.ones((n,), dtype=self.transport_matrix.dtype)

        import jax.numpy as jnp

        return jnp.ones((n,), dtype=self.transport_matrix.dtype)
