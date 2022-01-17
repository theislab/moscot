from typing import Tuple

from numpy import typing as npt
from ott.core.sinkhorn import SinkhornOutput as OTTSinkhornOutput
from ott.core.sinkhorn_lr import LRSinkhornOutput as OTTLRSinkhornOutput
from ott.core.gromov_wasserstein import GWOutput as OTTGWOutput
import jax.numpy as jnp

from moscot.solvers._output import MatrixSolverOutput, PotentialSolverOutput


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


class LRSinkhornOutput(SinkhornOutput):
    def __init__(self, output: OTTLRSinkhornOutput, *, threshold: float):
        super(SinkhornOutput, self).__init__(None, None)
        self._output = output
        self._threshold = threshold

    @property
    def shape(self) -> Tuple[int, int]:
        return self._output.geom.shape

    @property
    def converged(self) -> bool:
        costs, tol = self._output.costs, self._threshold
        costs = costs[costs != -1]
        # TODO(michalk8): is this correct?
        # modified the condition from:
        # https://github.com/google-research/ott/blob/a2be0c0703bd5b37cc0ef41e4c79bc10419ca542/ott/core/sinkhorn_lr.py#L239
        return bool(
            len(costs) > 1 and jnp.isfinite(costs[-1]) and jnp.isclose(costs[-2], costs[-1], rtol=self._threshold)
        )


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
