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


class ConvergencePlotterMixin:
    _NOT_COMPUTED = -1.0

    def __init__(self, costs: jnp.ndarray, errors: Optional[jnp.ndarray], *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # TODO(michalk8): don't plot costs?
        self._costs = costs[costs != self._NOT_COMPUTED]
        # TODO(michalk8): always compute errors?
        self._errors = None if errors is None else errors[errors != self._NOT_COMPUTED]

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
            How many of the last k steps of the algorithm to plot. If `None`, plot the full curve.
        title
            Title of the plot. If `None`, it is determined automatically.
        figsize
            Size of the figure.
        dpi
            Dots per inch.
        save
            Path where to save the figure.
        return_fig
            Whether to return the figure.
        kwargs
            Keyword arguments for :meth:`~matplotlib.axes.Axes.plot`.

        Returns
        -------
        The figure if ``return_fig = True``.
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
    """Output of various optimal transport problems.

    Parameters
    ----------
    output
        Output of the :mod:`ott` backend.
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
        return self._output.apply(x.T, axis=1 - forward).T  # convert to batch first

    @property
    def shape(self) -> Tuple[int, int]:
        if isinstance(self._output, OTTSinkhornOutput):
            return self._output.f.shape[0], self._output.g.shape[0]
        return self._output.geom.shape

    @property
    def transport_matrix(self) -> ArrayLike:
        return self._output.matrix

    def to(self, device: Optional[Device_t] = None) -> "OTTOutput":
        if isinstance(device, str) and ":" in device:
            device, ix = device.split(":")
            idx = int(ix)
        else:
            idx = 0

        if not isinstance(device, xla_ext.Device):
            try:
                device = jax.devices(device)[idx]
            except IndexError:
                raise IndexError(f"Unable to fetch the device with `id={idx}`.")

        return OTTOutput(jax.device_put(self._output, device))

    @property
    def cost(self) -> float:
        if isinstance(self._output, (OTTSinkhornOutput, OTTLRSinkhornOutput)):
            return float(self._output.reg_ot_cost)
        return float(self._output.reg_gw_cost)

    @property
    def converged(self) -> bool:
        return bool(self._output.converged)

    @property
    def potentials(self) -> Optional[Tuple[ArrayLike, ArrayLike]]:

        if isinstance(self._output, OTTSinkhornOutput):
            return self._output.f, self._output.g
        return None

    @property
    def rank(self) -> int:
        lin_output = self._output.linear_state if isinstance(self._output, OTTGWOutput) else self._output
        return len(lin_output.g) if isinstance(lin_output, OTTLRSinkhornOutput) else -1

    def _ones(self, n: int) -> jnp.ndarray:
        return jnp.ones((n,))
