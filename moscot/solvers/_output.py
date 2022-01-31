from abc import ABC, abstractmethod
from typing import Any, Tuple, Callable

import numpy.typing as npt


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

    def push(self, x: npt.ArrayLike) -> npt.ArrayLike:
        if x.shape[0] != self.shape[0]:
            raise ValueError("TODO: wrong shape")
        return self._apply(x, forward=True)

    def pull(self, x: npt.ArrayLike) -> npt.ArrayLike:
        if x.shape[0] != self.shape[1]:
            raise ValueError("TODO: wrong shape")
        return self._apply(x, forward=False)

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
        self._matrix = matrix

    def _apply(self, x: npt.ArrayLike, *, forward: bool) -> npt.ArrayLike:
        if forward:
            return self.transport_matrix.T @ x
        return self.transport_matrix @ x

    @property
    def transport_matrix(self) -> npt.ArrayLike:
        return self._matrix

    @property
    def shape(self) -> Tuple[int, int]:
        return self.transport_matrix.shape


class PotentialSolverOutput(BaseSolverOutput, ABC):
    def __init__(self, f: npt.ArrayLike, g: npt.ArrayLike):
        self._f = f
        self._g = g

    @property
    def shape(self) -> Tuple[int, int]:
        return self._f.shape[0], self._g.shape[0]
