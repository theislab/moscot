from abc import ABC, abstractmethod
from copy import copy
from typing import Any, Tuple, Callable, Iterable, Optional
from functools import partial

from scipy.sparse.linalg import LinearOperator

from moscot._types import Device_t, ArrayLike, DTypeLike
from moscot._logging import logger

__all__ = ["BaseSolverOutput", "MatrixSolverOutput"]


class BaseSolverOutput(ABC):
    @abstractmethod
    def _apply(self, x: ArrayLike, *, forward: bool) -> ArrayLike:
        pass

    @property
    @abstractmethod
    def transport_matrix(self) -> ArrayLike:
        pass

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int]:
        pass

    @property
    @abstractmethod
    def cost(self) -> float:
        pass

    @property
    @abstractmethod
    def converged(self) -> bool:
        pass

    @property
    @abstractmethod
    def potentials(self) -> Tuple[Optional[ArrayLike], Optional[ArrayLike]]:
        pass

    @abstractmethod
    def to(self, device: Optional[Device_t] = None) -> "BaseSolverOutput":
        pass

    @property
    def rank(self) -> int:
        return -1

    @property
    def is_low_rank(self) -> bool:
        return self.rank > -1

    # TODO(michalk8): mention in docs it needs to be broadcastable
    @abstractmethod
    def _ones(self, n: int) -> ArrayLike:
        pass

    def push(self, x: ArrayLike, scale_by_marginals: bool = False) -> ArrayLike:
        if x.ndim not in (1, 2):
            raise ValueError(f"Expected 1D or 2D array, found `{x.ndim}`.")
        if x.shape[0] != self.shape[0]:
            raise ValueError(f"Expected array to have shape `({self.shape[0]}, ...)`, found `{x.shape}`.")
        if scale_by_marginals:
            x = self._scale_by_marginals(x, forward=True)
        return self._apply(x, forward=True)

    def pull(self, x: ArrayLike, scale_by_marginals: bool = False) -> ArrayLike:
        if x.ndim not in (1, 2):
            raise ValueError(f"Expected 1D or 2D array, found `{x.ndim}`.")
        if x.shape[0] != self.shape[1]:
            raise ValueError(f"Expected array to have shape `({self.shape[1]}, ...)`, found `{x.shape}`.")
        if scale_by_marginals:
            x = self._scale_by_marginals(x, forward=False)
        return self._apply(x, forward=False)

    def as_linear_operator(self, *, forward: bool, scale_by_marginals: bool = False) -> LinearOperator:
        push = partial(self.push, scale_by_marginals=scale_by_marginals)
        pull = partial(self.pull, scale_by_marginals=scale_by_marginals)
        mv, rmv = (pull, push) if forward else (push, pull)  # please do not change this line
        return LinearOperator(shape=self.shape, dtype=self.a.dtype, matvec=mv, rmatvec=rmv)

    def chain(
        self, outputs: Iterable["BaseSolverOutput"], forward: bool, scale_by_marginals: bool = False
    ) -> LinearOperator:
        op = self.as_linear_operator(forward=forward, scale_by_marginals=scale_by_marginals)
        for out in outputs:
            op *= out.as_linear_operator(forward=forward, scale_by_marginals=scale_by_marginals)

        return op

    @property
    def a(self) -> ArrayLike:
        """
        Marginals of the source distribution.

        If output of an unbalanced OT problem, these are the posterior marginals.
        """
        return self.pull(self._ones(self.shape[1]))

    @property
    def b(self) -> ArrayLike:
        """
        Marginals of the target distribution.

        If output of an unbalanced OT problem, these are the posterior marginals.
        """
        return self.push(self._ones(self.shape[0]))

    @property
    def dtype(self) -> DTypeLike:
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


class MatrixSolverOutput(BaseSolverOutput, ABC):  # noqa: B024
    def __init__(self, matrix: ArrayLike):
        super().__init__()
        self._matrix = matrix

    def _apply(self, x: ArrayLike, *, forward: bool) -> ArrayLike:
        if forward:
            return self.transport_matrix.T @ x
        return self.transport_matrix @ x

    @property
    def transport_matrix(self) -> ArrayLike:
        """%(transport_matrix)s"""  # noqa: D400
        return self._matrix

    @property
    def shape(self) -> Tuple[int, int]:
        """%(shape)s"""  # noqa: D400
        return self.transport_matrix.shape  # type: ignore[return-value]

    def to(self, device: Optional[Device_t] = None, dtype: Optional[DTypeLike] = None) -> "BaseSolverOutput":
        if device is not None:
            logger.info(f"`{self.__class__.__name__}` doesn't support `device` argument. Ignoring")
            return self
        if dtype is None:
            return self

        obj = copy(self)
        obj._matrix = obj.transport_matrix.astype(dtype)
        return obj
