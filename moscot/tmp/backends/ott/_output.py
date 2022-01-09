from numpy import typing as npt
from ott.core.sinkhorn import SinkhornOutput as OTTSinkhornOutput
from ott.core.gromov_wasserstein import GWOutput as OTTGWOutput

from moscot.tmp.solvers._output import MatrixSolverOutput, PotentialSolverOutput


class SinkhornOutput(PotentialSolverOutput):
    def __init__(self, output: OTTSinkhornOutput):
        super().__init__(output.f, output.g)
        self._output = output

    def _apply(self, x: npt.ArrayLike, *, forward: bool) -> npt.ArrayLike:
        axis = int(not forward)
        if x.ndim == 1:
            return self._output.apply(x, axis=axis)
        if x.ndim == 2:
            # convert to batch first
            return self._output.apply(x.T, axis=axis).T

        raise ValueError("TODO - dim error")

    @property
    def transport_matrix(self) -> npt.ArrayLike:
        return self._output.matrix

    @property
    def cost(self) -> float:
        return float(self._output.reg_ot_cost)

    @property
    def converged(self) -> bool:
        return bool(self._output.converged)


class GWOutput(MatrixSolverOutput):
    def __init__(self, output: OTTGWOutput):
        super().__init__(output.transport)
        self._converged = bool(output.convergence)
        self._cost = float(output.reg_gw_cost)

    @property
    def cost(self) -> float:
        return self._cost

    @property
    def converged(self) -> bool:
        return self._converged
