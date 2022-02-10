from abc import ABC, abstractmethod
from typing import Any, Tuple, Callable

import numpy.typing as npt


class BaseSolverOutput(ABC):
    @abstractmethod
    def _apply(self, x: npt.ArrayLike, *, forward: bool, scale_by_marginals: bool = False) -> npt.ArrayLike:
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

    def push(self, x: npt.ArrayLike, scale_by_marginals: bool = False) -> npt.ArrayLike:
        if x.shape[0] != self.shape[0]:
            raise ValueError("TODO: wrong shape")
        return self._apply(x, forward=True, scale_by_marginals=scale_by_marginals)

    def pull(self, x: npt.ArrayLike, scale_by_marginals: bool = False) -> npt.ArrayLike:
        if x.shape[0] != self.shape[1]:
            raise ValueError("TODO: wrong shape")
        return self._apply(x, forward=False, scale_by_marginals=scale_by_marginals)

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

    def _apply(self, x: npt.ArrayLike, *, forward: bool, scale_by_marginals: bool = False) -> npt.ArrayLike:
        if scale_by_marginals:
            raise NotImplementedError("TODO")
        if forward:
            return self.transport_matrix.T @ x
        return self.transport_matrix @ x

    @property
    def transport_matrix(self) -> npt.ArrayLike:
        return self._matrix

    @property
    def shape(self) -> Tuple[int, int]:
        return self.transport_matrix.shape
