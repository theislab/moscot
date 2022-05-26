from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Tuple, Callable, Iterable

import numpy as np
from scipy.sparse.linalg import LinearOperator

from moscot._types import ArrayLike

# TODO(michalk8):
#  1. mb. use more contrained type hints
#  2. consider always returning 2-dim array, even if 1-dim is passed (not sure which convenient for user)

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
    def potentials(self) -> Tuple[ArrayLike, ArrayLike]:
        pass

    @property
    def rank(self) -> int:
        return -1

    # TODO(michalk8): mention in docs it needs to be broadcastable
    @abstractmethod
    def _ones(self, n: int) -> ArrayLike:
        pass

    def push(self, x: ArrayLike, scale_by_marginals: bool = False) -> ArrayLike:
        if x.shape[0] != self.shape[0]:
            raise ValueError("TODO: wrong shape")
        x = self._scale_by_marginals(x, forward=True) if scale_by_marginals else x
        return self._apply(x, forward=True)

    def pull(self, x: ArrayLike, scale_by_marginals: bool = False) -> ArrayLike:
        if x.shape[0] != self.shape[1]:
            raise ValueError("TODO: wrong shape")
        x = self._scale_by_marginals(x, forward=False) if scale_by_marginals else x
        return self._apply(x, forward=False)

    @property
    def a(self) -> ArrayLike:
        """Marginals of source distribution. If output of unbalanced OT, these are the posterior marginals."""
        return self.pull(self._ones(self.shape[1]))

    @property
    def b(self) -> ArrayLike:
        """Marginals of target distribution. If output of unbalanced OT, these are the posterior marginals."""
        return self.push(self._ones(self.shape[0]))

    @property
    def dtype(self) -> Any:  # TODO(michalk8): typeme
        return self.a.dtype

    def as_linear_operator(self, *, forward: bool, scale_by_marginals: bool = False) -> LinearOperator:
        push = partial(self.push, scale_by_marginals=scale_by_marginals)
        pull = partial(self.pull, scale_by_marginals=scale_by_marginals)
        mv, rmv = (push, pull) if forward else (pull, push)
        return LinearOperator(shape=self.shape, dtype=self.a.dtype, matvec=mv, rmatvec=rmv)

    def chain(
        self, outputs: Iterable["BaseSolverOutput"], forward: bool, scale_by_marginals: bool = False
    ) -> LinearOperator:
        op = self.as_linear_operator(forward=forward, scale_by_marginals=scale_by_marginals)
        for out in outputs:
            op *= out.as_linear_operator(forward=forward, scale_by_marginals=scale_by_marginals)

        return op

    def _scale_by_marginals(self, x: ArrayLike, *, forward: bool) -> ArrayLike:
        # alt. we could use the public push/pull
        marginals = self.a if forward else self.b
        if x.ndim == 2:
            marginals = marginals[:, None]
        return x / (marginals + 1e-12)

    # TODO(michalk8): the below are not efficient (+1 is redundant)
    def _scale_transport_by_marginals(self, forward: bool) -> ArrayLike:
        if forward:
            scaled_transport = np.dot(np.diag(1 / self.a), self.transport_matrix)
        else:
            scaled_transport = np.dot(self.transport_matrix, np.diag(1 / self.b))
        return scaled_transport

    def _scale_transport_by_sum(self, forward: bool) -> ArrayLike:
        if forward:
            scaled_transport = self.transport_matrix / self.transport_matrix.sum(1)[:, None]
        else:
            scaled_transport = self.transport_matrix / self.transport_matrix.sum(0)[None, :]
        return scaled_transport

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
        return self.transport_matrix.shape

    @property
    def potentials(self):  # TODO(michalk8): refactor
        raise NotImplementedError("This solver does not allow for potentials")
