from abc import ABC
from typing import Any, Tuple, Union, Optional

from matplotlib.figure import Figure
from typing_extensions import Literal
import matplotlib.pyplot as plt

from ott.core.sinkhorn import SinkhornOutput as OTTSinkhornOutput
from ott.core.sinkhorn_lr import LRSinkhornOutput as OTTLRSinkhornOutput
from ott.core.gromov_wasserstein import GWOutput as OTTGWOutput
import jax.numpy as jnp

from moscot._types import ArrayLike
from moscot.solvers._output import HasPotentials, BaseSolverOutput, MatrixSolverOutput

__all__ = ["LinearOutput", "LRLinearOutput", "QuadraticOutput"]


class RankMixin:
    def __init__(self, *args: Any, rank: int, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._rank = max(-1, rank)

    @property
    def rank(self) -> int:
        return self._rank


class CostMixin:
    NOT_COMPUTED = -1.0

    def __init__(self, costs: jnp.ndarray, errors: Optional[jnp.ndarray], *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._costs = costs[costs != self.NOT_COMPUTED]
        self._errors = None if errors is None else errors[errors != self.NOT_COMPUTED]

    @property
    def cost(self) -> float:
        """TODO."""
        return float(self._costs[-1])

    def plot_convergence(
        self,
        kind: Literal["error", "cost"] = "cost",
        title: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        save: Optional[str] = None,
        return_fig: bool = False,
        **kwargs: Any,
    ) -> Optional[Figure]:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        if kind == "error":
            values = self._errors
        elif kind == "cost":
            values = self._costs
        else:
            raise ValueError(f"TODO: invalid `kind={kind}`")
        if values is None:
            raise ValueError(f"TODO: no data for `kind={kind}`")

        ax.plot(values, **kwargs)
        ax.set_xlabel("iteration")
        ax.set_ylabel(kind)
        if title is None:
            title = "converged" if self.converged else "not converged"  # type: ignore[attr-defined]
        ax.set_title(title)

        if save is not None:
            fig.savefig(save)
        if return_fig:
            return fig


class OTTOutput(CostMixin, BaseSolverOutput, ABC):
    def __init__(self, output: Union[OTTSinkhornOutput, OTTLRSinkhornOutput], **_: Any):
        if isinstance(output, OTTSinkhornOutput):
            costs, errors = jnp.asarray([output.reg_ot_cost]), output.errors
        else:
            costs, errors = output.costs, None
        super().__init__(costs=costs, errors=errors)
        self._output = output

    @property
    def transport_matrix(self) -> ArrayLike:
        """%(transport_matrix)s."""
        return self._output.matrix

    @property
    def converged(self) -> bool:
        """%(converged)s."""
        return bool(self._output.converged)

    def _ones(self, n: int) -> jnp.ndarray:
        return jnp.ones((n,))


class LinearOutput(HasPotentials, OTTOutput):
    """Output class for linear OT problems."""

    def _apply(self, x: ArrayLike, *, forward: bool) -> ArrayLike:
        if x.ndim == 1:
            return self._output.apply(x, axis=1 - forward)
        if x.ndim == 2:
            # convert to batch first
            return self._output.apply(x.T, axis=1 - forward).T
        raise ValueError("TODO - dim error")

    def plot_convergence(self, *args: Any, **kwargs: Any) -> Optional[Figure]:
        return super().plot_convergence("error", *args[1:], **kwargs)

    @property
    def shape(self) -> Tuple[int, int]:
        """%(shape)s."""
        return self._output.f.shape[0], self._output.g.shape[0]

    @property
    def potentials(self) -> Tuple[ArrayLike, ArrayLike]:
        """Potentials obtained from Sinkhorn algorithm."""
        return self._output.f, self._output.g


class LRLinearOutput(RankMixin, OTTOutput):
    """Output class for low-rank linear OT problems."""

    def _apply(self, x: ArrayLike, *, forward: bool) -> ArrayLike:
        axis = int(not forward)
        if x.ndim == 1:
            return self._output.apply(x, axis=axis)
        if x.ndim == 2:
            return jnp.stack([self._output.apply(x_, axis=axis) for x_ in x.T]).T
        raise ValueError("TODO - dim error")

    @property
    def shape(self) -> Tuple[int, int]:
        """%(shape)s."""
        return self._output.geom.shape


class QuadraticOutput(CostMixin, RankMixin, MatrixSolverOutput):
    """
    Output class for Gromov-Wasserstein problems.

    This class wraps :class:`ott.core.gromov_wasserstein.QuadraticOutput`.

    Parameters
    ----------
    output
        Instance of :class:`ott.core.gromov_wasserstein.QuadraticOutput`.
    rank
        Rank of the solver. `-1` if full-rank was used.
    """

    def __init__(self, output: OTTGWOutput, *, rank: int = -1):
        super().__init__(output.costs, output.errors, output.matrix, rank=rank)
        self._converged = bool(output.convergence)

    @property
    def converged(self) -> bool:
        """%(converged)s"""
        return self._converged

    def _ones(self, n: int) -> jnp.ndarray:
        return jnp.ones((n,))
