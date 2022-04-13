from abc import ABC
from typing import Any, Tuple, Union

from numpy import typing as npt
from ott.core.sinkhorn import SinkhornOutput as OTTSinkhornOutput
from ott.core.sinkhorn_lr import LRSinkhornOutput as OTTLRSinkhornOutput
from ott.core.gromov_wasserstein import GWOutput as OTTGWOutput
import jax.numpy as jnp

from moscot.solvers._output import BaseSolverOutput, MatrixSolverOutput

__all__ = ("SinkhornOutput", "LRSinkhornOutput", "GWOutput")


class OutputRankMixin:
    def __init__(self, *args: Any, rank: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._rank = max(-1, rank)

    @property
    def rank(self) -> int:
        return self._rank


class LinearOTTOutput(BaseSolverOutput, ABC):
    def __init__(self, output: Union[OTTSinkhornOutput, OTTLRSinkhornOutput], **_: Any):
        super().__init__()
        self._output = output

    @property
    def transport_matrix(self) -> npt.ArrayLike:
        return self._output.matrix

    @property
    def cost(self) -> float:
        return float(self._output.reg_ot_cost)

    @property
    def converged(self) -> bool:
        return bool(self._output.converged)

    def _ones(self, n: int) -> jnp.ndarray:
        return jnp.ones((n,))


class SinkhornOutput(LinearOTTOutput):
    def _apply(self, x: npt.ArrayLike, *, forward: bool) -> npt.ArrayLike:
        if x.ndim == 1:
            return self._output.apply(x, axis=1 - forward)
        if x.ndim == 2:
            # convert to batch first
            return self._output.apply(x.T, axis=1 - forward).T
        raise ValueError("TODO - dim error")

    @property
    def shape(self) -> Tuple[int, int]:
        return self._output.f.shape[0], self._output.g.shape[0]

    @property
    def potentials(self) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        return self._output.f, self._output.g


class LRSinkhornOutput(OutputRankMixin, LinearOTTOutput):
    def _apply(self, x: npt.ArrayLike, *, forward: bool) -> npt.ArrayLike:
        axis = int(not forward)
        if x.ndim == 1:
            return self._output.apply(x, axis=axis)
        if x.ndim == 2:
            return jnp.stack([self._output.apply(x_, axis=axis) for x_ in x.T]).T
        raise ValueError("TODO - dim error")

    @property
    def shape(self) -> Tuple[int, int]:
        return self._output.geom.shape

    @property
    def potentials(self):
        raise NotImplementedError("This solver does not allow for potentials")


class GWOutput(OutputRankMixin, MatrixSolverOutput):
    def __init__(self, output: OTTGWOutput, rank: int = -1):
        super().__init__(output.matrix, rank=rank)
        self._converged = bool(output.convergence)
        self._cost = float(output.reg_gw_cost)

    @property
    def cost(self) -> float:
        return self._cost

    @property
    def converged(self) -> bool:
        return self._converged

    def _ones(self, n: int) -> jnp.ndarray:
        return jnp.ones((n,))
