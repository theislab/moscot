from abc import ABC
from typing import Any, Tuple, Union, Iterator, Optional
import contextlib

from matplotlib.figure import Figure
from typing_extensions import Literal
import matplotlib.pyplot as plt

from ott.core.sinkhorn import SinkhornOutput as OTTSinkhornOutput
from ott.core.sinkhorn_lr import LRSinkhornOutput as OTTLRSinkhornOutput
from ott.core.gromov_wasserstein import GWOutput as OTTGWOutput
import jax
import jax.numpy as jnp
import jaxlib.xla_extension as xla_ext

from moscot._types import ArrayLike, DTypeLike
from moscot.solvers._output import BaseSolverOutput

__all__ = ["OTTOutput"]

Device_t = Union[str, xla_ext.Device, Literal["cpu", "gpu", "tpu"]]


@contextlib.contextmanager
def enable_x64() -> Iterator[None]:
    old_value = jax.config.jax_enable_x64  # type: ignore[attr-defined]
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old_value)


class RankMixin:
    def __init__(self, *args: Any, rank: int, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._rank = max(-1, rank)

    @property
    def rank(self) -> int:
        return self._rank


class ConvergencePlotterMixin:
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
        last_k: Optional[int] = None,
        title: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        save: Optional[str] = None,
        return_fig: bool = False,
        **kwargs: Any,
    ) -> Optional[Figure]:
        """
        Plot the convergence curve.

        Parameters
        ----------
        last_k
            How many of the last k steps of the algorithm to plot. If `None`, the full curve is plotted
        %(plotting_title)
        %(plotting)s
        """

        def select_values(last_k: Optional[int] = None) -> Tuple[str, jnp.ndarray, jnp.ndarray]:
            # `> 1` because of pure Sinkhorn
            if len(self._costs) > 1 or self._errors is None:
                last_k = min(last_k, len(self._costs)) if last_k is not None else len(self._costs)
                return "cost", self._costs[-last_k:], range(len(self._costs))[-last_k:]
            last_k = min(last_k, len(self._errors)) if last_k is not None else len(self._errors)
            return "error", self._errors[-last_k:], range(len(self._errors))[-last_k:]

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        kind, values, xs = select_values(last_k)

        ax.plot(xs, values, **kwargs)
        ax.set_xlabel("iteration")
        ax.set_ylabel(kind)
        if title is None:
            title = "converged" if self.converged else "not converged"  # type: ignore[attr-defined]
        ax.set_title(title)

        if save is not None:
            fig.savefig(save)
        if return_fig:
            return fig


class OTTOutput(RankMixin, ConvergencePlotterMixin, BaseSolverOutput, ABC):
    """
    Output representation of various OT problems.

    Parameters
    ----------
    output
        One of:

            - :class:`ott.core.sinkhorn.SinkhornOutput`.
            - :class:`ott.core.sinkhorn_lr.LRSinkhornOutput`.
            - :class:`ott.core.gromov_wasserstein.QuadraticOutput`.
    rank
        Rank of the solver. `-1` if full-rank was used.
    """

    def __init__(self, output: Union[OTTSinkhornOutput, OTTLRSinkhornOutput, OTTGWOutput], rank: int = -1):
        # TODO(michalk8): think about whether we want to plot the error in inner Sinkhorn in GW
        if isinstance(output, OTTSinkhornOutput):
            costs, errors = jnp.asarray([output.reg_ot_cost]), output.errors
        else:
            costs, errors = output.costs, None
        super().__init__(rank=rank, costs=costs, errors=errors)
        self._output = output

    def _apply(self, x: ArrayLike, *, forward: bool) -> ArrayLike:
        if x.ndim == 1:
            return self._output.apply(x, axis=1 - forward)
        if x.ndim == 2:
            return self._output.apply(x.T, axis=1 - forward).T  # batch first
        raise ValueError("TODO - dim error")

    @property
    def shape(self) -> Tuple[int, int]:
        """%(shape)s."""
        # TODO(michalk8): add to OTT
        if isinstance(self._output, OTTSinkhornOutput):
            return self._output.f.shape[0], self._output.g.shape[0]
        return self._output.geom.shape

    @property
    def transport_matrix(self) -> ArrayLike:
        """%(transport_matrix)s."""
        return self._output.matrix

    def to(
        self,
        device: Optional[Device_t] = None,
        dtype: Optional[DTypeLike] = None,
    ) -> "OTTOutput":
        """
        Transfer the output to another device or change its data type.

        Parameters
        ----------
        device
            Device where to transfer the solver output.
        dtype
            Data type of underlying arrays.

        Returns
        -------
        Self with possibly modified device and dtypes.
        """
        # TODO(michalk8): when polishing docs, move the definition to the base class + use docrep
        def convert_dtype(val: Any) -> Any:
            return val.astype(dtype) if isinstance(val, jnp.ndarray) else val

        ix = 0
        if isinstance(device, str) and ":" in device:
            device, ix = device.split(":")  # type: ignore[assignment]
            ix = int(ix)

        if not isinstance(device, xla_ext.Device):
            try:
                device = jax.devices(device)[ix]
            except IndexError:
                raise IndexError("TODO: indexing error when fetching device") from None

        out = jax.device_put(self._output, device)
        with enable_x64():
            out = jax.tree_map(convert_dtype, out)  # type: ignore[attr-defined]
        return OTTOutput(out, rank=self.rank)

    @property
    def converged(self) -> bool:
        """%(converged)s."""
        # TODO(michalk8): unify in OTT
        if isinstance(self._output, OTTGWOutput):
            return bool(self._output.convergence)
        return bool(self._output.converged)

    @property
    def potentials(self) -> Tuple[Optional[ArrayLike], Optional[ArrayLike]]:
        """Potentials obtained from Sinkhorn algorithm."""
        if isinstance(self._output, OTTSinkhornOutput):
            return self._output.f, self._output.g
        return None, None

    def _ones(self, n: int) -> jnp.ndarray:
        return jnp.ones((n,))
