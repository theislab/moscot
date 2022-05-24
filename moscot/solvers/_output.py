from abc import ABC, abstractmethod
from typing import Any, Tuple, Callable

import numpy as np
import numpy.typing as npt

# TODO(michalk8):
#  1. mb. use more contrained type hints
#  2. consider always returning 2-dim array, even if 1-dim is passed (not sure which convenient for user)

__all__ = ["BaseSolverOutput", "MatrixSolverOutput", "JointOperator"]


class BaseSolverOutput(ABC):
    @abstractmethod
    def _apply(self, x: npt.ArrayLike, *, forward: bool) -> npt.ArrayLike:
        pass

    @property
    @abstractmethod
    def transport_matrix(self) -> npt.ArrayLike:
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
    def potentials(self) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        pass

    @property
    def rank(self) -> int:
        return -1

    # TODO(michalk8): mention in docs it needs to be broadcastable
    @abstractmethod
    def _ones(self, n: int) -> npt.ArrayLike:
        pass

    def push(self, x: npt.ArrayLike, scale_by_marginals: bool = False) -> npt.ArrayLike:
        if x.shape[0] != self.shape[0]:
            raise ValueError("TODO: wrong shape")
        x = self._scale_by_marginals(x, forward=True) if scale_by_marginals else x
        return self._apply(x, forward=True)

    def pull(self, x: npt.ArrayLike, scale_by_marginals: bool = False) -> npt.ArrayLike:
        if x.shape[0] != self.shape[1]:
            raise ValueError("TODO: wrong shape")
        x = self._scale_by_marginals(x, forward=False) if scale_by_marginals else x
        return self._apply(x, forward=False)

    @property
    def a(self) -> npt.ArrayLike:
        """Marginals of source distribution. If output of unbalanced OT, these are the posterior marginals."""
        return self.pull(self._ones(self.shape[1]))

    @property
    def b(self) -> npt.ArrayLike:
        """Marginals of target distribution. If output of unbalanced OT, these are the posterior marginals."""
        return self.push(self._ones(self.shape[0]))

    def _scale_by_marginals(self, x: npt.ArrayLike, *, forward: bool) -> npt.ArrayLike:
        # alt. we could use the public push/pull
        marginals = self.a if forward else self.b
        if x.ndim == 2:
            marginals = marginals[:, None]
        return x / (marginals + 1e-12)

    def _scale_transport_by_marginals(self, forward: bool) -> npt.ArrayLike:
        if forward:
            scaled_transport = np.dot(np.diag(1 / self.a), self.transport_matrix)
        else:
            scaled_transport = np.dot(self.transport_matrix, np.diag(1 / self.b))
        return scaled_transport

    def _scale_transport_by_sum(self, forward: bool) -> npt.ArrayLike:
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
    def __init__(self, matrix: npt.ArrayLike):
        super().__init__()
        self._matrix = matrix

    def _apply(self, x: npt.ArrayLike, *, forward: bool) -> npt.ArrayLike:
        if forward:
            return self.transport_matrix.T @ x
        return self.transport_matrix @ x

    @property
    def transport_matrix(self) -> npt.ArrayLike:
        """%(transport_matrix)s"""  # noqa: D400
        return self._matrix

    @property
    def shape(self) -> Tuple[int, int]:
        """%(shape)s"""  # noqa: D400
        return self.transport_matrix.shape

    @property
    def potentials(self):  # TODO(michalk8): refactor
        raise NotImplementedError("This solver does not allow for potentials")


class JointOperator:
    def __init__(self, outputs: Tuple[BaseSolverOutput, ...]):
        if not len(outputs):
            raise ValueError("TODO: no solver outputs")
        for curr, next in zip(outputs[:-1], outputs[1:]):  # noqa: A001
            if curr.shape[1] != next.shape[0]:
                raise ValueError("TODO: outputs shape mismatch")

        self._outputs = outputs

    @property
    def shape(self) -> Tuple[int, int]:
        return self._outputs[0].shape[0], self._outputs[-1].shape[1]

    def push(self, x: npt.ArrayLike, **kwargs: Any) -> npt.ArrayLike:
        for op in self._outputs:
            x = op.push(x, **kwargs)
        return x

    def pull(self, x: npt.ArrayLike, **kwargs: Any) -> npt.ArrayLike:
        for op in reversed(self._outputs):
            x = op.pull(x, **kwargs)
        return x
