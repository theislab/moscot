from typing import Any, Tuple, Union, Optional

from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from ott.core.sinkhorn import SinkhornOutput as OTTSinkhornOutput
from ott.core.sinkhorn_lr import LRSinkhornOutput as OTTLRSinkhornOutput
from ott.core.gromov_wasserstein import GWOutput as OTTGWOutput
import jax
import jax.numpy as jnp
import jaxlib.xla_extension as xla_ext

from moscot._types import Device_t, ArrayLike
from moscot.solvers._output import BaseSolverOutput

__all__ = ["OTTOutput"]


# TODO(michalk8): merge to OTTOutput
class ConvergencePlotterMixin:
    NOT_COMPUTED = -1.0

    def __init__(self, costs: jnp.ndarray, errors: Optional[jnp.ndarray], *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._costs = costs[costs != self.NOT_COMPUTED]
        self._errors = None if errors is None else errors[errors != self.NOT_COMPUTED]

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
            How many of the last k steps of the algorithm to plot. If `None`, the full curve is plotted.

        Returns
        -------
        TODO.
        """

        def select_values(last_k: Optional[int] = None) -> Tuple[str, jnp.ndarray, jnp.ndarray]:
            # `> 1` because of pure Sinkhorn
            if len(self._costs) > 1 or self._errors is None:
                metric = self._costs
                metric_str = "cost"
            else:
                metric = self._errors
                metric_str = "error"
            last_k = min(last_k, len(metric)) if last_k is not None else len(metric)
            return metric_str, metric[-last_k:], range(len(metric))[-last_k:]

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


class OTTOutput(ConvergencePlotterMixin, BaseSolverOutput):
    """
    Output representation of various OT problems.

    Parameters
    ----------
    output
        One of:

            - :class:`ott.core.sinkhorn.SinkhornOutput`.
            - :class:`ott.core.sinkhorn_lr.LRSinkhornOutput`.
            - :class:`ott.core.gromov_wasserstein.QuadraticOutput`.
    """

    def __init__(self, output: Union[OTTSinkhornOutput, OTTLRSinkhornOutput, OTTGWOutput]):
        # TODO(michalk8): think about whether we want to plot the error in inner Sinkhorn in GW
        if isinstance(output, OTTSinkhornOutput):
            costs, errors = jnp.asarray([output.reg_ot_cost]), output.errors
        else:
            costs, errors = output.costs, None
        super().__init__(costs=costs, errors=errors)
        self._output = output

    def _apply(self, x: ArrayLike, *, forward: bool) -> ArrayLike:
        if x.ndim == 1:
            return self._output.apply(x, axis=1 - forward)
        if x.ndim == 2:
            return self._output.apply(x.T, axis=1 - forward).T  # batch first
        raise ValueError("TODO - dim error")

    @property
    def shape(self) -> Tuple[int, int]:
        """%(shape)s"""
        # TODO(michalk8): add to OTT
        if isinstance(self._output, OTTSinkhornOutput):
            return self._output.f.shape[0], self._output.g.shape[0]
        return self._output.geom.shape

    @property
    def transport_matrix(self) -> ArrayLike:
        """%(transport_matrix)s"""
        return self._output.matrix

    def to(
        self,
        device: Optional[Device_t] = None,
    ) -> "OTTOutput":
        """
        Transfer the output to another device or change its data type.

        Parameters
        ----------
        device
            Device where to transfer the solver output.

        Returns
        -------
        Self with possibly modified device and dtypes.
        """
        # TODO(michalk8): when polishing docs, move the definition to the base class + use docrep
        if isinstance(device, str) and ":" in device:
            device, ix = device.split(":")
            idx = int(ix)
        else:
            idx = 0

        if not isinstance(device, xla_ext.Device):
            try:
                device = jax.devices(device)[idx]
            except IndexError:
                raise IndexError("TODO: indexing error when fetching device") from None

        out = jax.device_put(self._output, device)
        return OTTOutput(out)

    @property
    def cost(self) -> float:
        """TODO."""
        if isinstance(self._output, (OTTSinkhornOutput, OTTLRSinkhornOutput)):
            return float(self._output.reg_ot_cost)
        return float(self._output.reg_gw_cost)

    @property
    def converged(self) -> bool:
        """%(converged)s."""
        return bool(self._output.converged)

    @property
    def potentials(self) -> Tuple[Optional[ArrayLike], Optional[ArrayLike]]:
        """Potentials obtained from Sinkhorn algorithm."""
        if isinstance(self._output, OTTSinkhornOutput):
            return self._output.f, self._output.g
        return None, None

    @property
    def rank(self) -> int:
        """Rank of the transport matrix."""
        lin_output = self._output.linear_state if isinstance(self._output, OTTGWOutput) else self._output
        return len(lin_output.g) if isinstance(lin_output, OTTLRSinkhornOutput) else -1

    def _ones(self, n: int) -> jnp.ndarray:
        return jnp.ones((n,))
