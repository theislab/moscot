from ot.backend import get_backend

from numpy import typing as npt

from moscot.solvers._output import MatrixSolverOutput

__all__ = ("POTOutput",)


class POTOutput(MatrixSolverOutput):
    def __init__(self, matrix: npt.ArrayLike, *, cost: float, converged: bool):
        super().__init__(matrix)
        self._cost = cost
        self._converged = converged

    @property
    def cost(self) -> float:
        return self._cost

    @property
    def converged(self) -> bool:
        return self._converged

    def _ones(self, n: int) -> npt.ArrayLike:
        nx = get_backend(self.transport_matrix)
        return nx.ones((n,))
