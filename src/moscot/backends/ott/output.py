from typing import Any, Optional, Tuple, Union

import jaxlib.xla_extension as xla_ext

import jax
import jax.numpy as jnp
import numpy as np
from ott.solvers.linear.sinkhorn import SinkhornOutput as OTTSinkhornOutput
from ott.solvers.linear.sinkhorn_lr import LRSinkhornOutput as OTTLRSinkhornOutput
from ott.solvers.quadratic.gromov_wasserstein import GWOutput as OTTGWOutput

import matplotlib as mpl
import matplotlib.pyplot as plt

from moscot._docs._docs import d
from moscot._types import ArrayLike, Device_t
from moscot.base.output import BaseSolverOutput

__all__ = ["OTTOutput"]


class OTTOutput(BaseSolverOutput):
    """Output of various optimal transport problems.

    Parameters
    ----------
    output
        Output of :mod:`ott` backend.
    """

    _NOT_COMPUTED = -1.0

    def __init__(self, output: Union[OTTSinkhornOutput, OTTLRSinkhornOutput, OTTGWOutput]):
        super().__init__()
        self._output = output
        self._costs = None if isinstance(output, OTTSinkhornOutput) else output.costs
        self._errors = output.errors

    @d.get_sections(base="plot_costs", sections=["Parameters", "Returns"])
    def plot_costs(
        self,
        last: Optional[int] = None,
        title: Optional[str] = None,
        return_fig: bool = False,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        save: Optional[str] = None,
        ax: Optional[mpl.axes.Axes] = None,
        **kwargs: Any,
    ) -> Optional[mpl.figure.Figure]:
        """Plot regularized OT costs during the iterations.

        Parameters
        ----------
        last
            How many of the last steps of the algorithm to plot. If `None`, plot the full curve.
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
        ax
            Axes on which to plot.
        kwargs
            Keyword arguments for :meth:`~matplotlib.axes.Axes.plot`.

        Returns
        -------
        The figure if ``return_fig = True``.
        """
        if self._costs is None:
            raise RuntimeError("No costs to plot.")

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi) if ax is None else (ax.get_figure(), ax)
        self._plot_lines(ax, np.asarray(self._costs), last=last, y_label="cost", title=title, **kwargs)

        if save is not None:
            fig.savefig(save)
        return fig if return_fig else None

    @d.dedent
    def plot_errors(
        self,
        last: Optional[int] = None,
        title: Optional[str] = None,
        outer_iteration: int = -1,
        return_fig: bool = False,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        save: Optional[str] = None,
        ax: Optional[mpl.axes.Axes] = None,
        **kwargs: Any,
    ) -> Optional[mpl.figure.Figure]:
        """Plot errors during the iterations.

        Parameters
        ----------
        %(plot_costs.parameters)s

        Returns
        -------
        %(plot_costs.returns)s
        """
        if self._errors is None:
            raise RuntimeError("No errors to plot.")

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi) if ax is None else (ax.get_figure(), ax)
        errors = np.asarray(self._errors)
        errors = errors[np.where(errors != self._NOT_COMPUTED)[0]]

        if not self.is_linear:  # handle Gromov's inner iterations
            if self._errors.ndim != 2:
                raise ValueError(f"Expected `errors` to be 2 dimensional array, found `{self._errors.ndim}`.")
            # convert to numpy because of how JAX handles indexing
            errors = errors[outer_iteration]

        self._plot_lines(ax, errors, last=last, y_label="error", title=title, **kwargs)

        if save is not None:
            fig.savefig(save)
        return fig if return_fig else None

    def _plot_lines(
        self,
        ax: mpl.axes.Axes,
        values: ArrayLike,
        last: Optional[int] = None,
        y_label: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if values.ndim != 1:
            raise ValueError(f"Expected array to be 1 dimensional, found `{values.ndim}`.")
        values = values[values != self._NOT_COMPUTED]
        ixs = np.arange(len(values))
        if last is not None:
            values = values[-last:]
            ixs = ixs[-last:]

        ax.plot(ixs, values, **kwargs)
        ax.set_xlabel("iteration (logged)")
        ax.set_ylabel(y_label)
        ax.set_title(title if title is not None else "converged" if self.converged else "not converged")
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

    def _apply(self, x: ArrayLike, *, forward: bool) -> ArrayLike:
        if x.ndim == 1:
            return self._output.apply(x, axis=1 - forward)
        return self._output.apply(x.T, axis=1 - forward).T  # convert to batch first

    @property
    def shape(self) -> Tuple[int, int]:  # noqa: D102
        if isinstance(self._output, OTTSinkhornOutput):
            return self._output.f.shape[0], self._output.g.shape[0]
        return self._output.geom.shape

    @property
    def transport_matrix(self) -> ArrayLike:  # noqa: D102
        return self._output.matrix

    @property
    def is_linear(self) -> bool:  # noqa: D102
        return isinstance(self._output, (OTTSinkhornOutput, OTTLRSinkhornOutput))

    def to(self, device: Optional[Device_t] = None) -> "OTTOutput":  # noqa: D102
        if isinstance(device, str) and ":" in device:
            device, ix = device.split(":")
            idx = int(ix)
        else:
            idx = 0

        if not isinstance(device, xla_ext.Device):
            try:
                device = jax.devices(device)[idx]
            except IndexError:
                raise IndexError(f"Unable to fetch the device with `id={idx}`.") from None

        return OTTOutput(jax.device_put(self._output, device))

    @property
    def cost(self) -> float:  # noqa: D102
        if isinstance(self._output, (OTTSinkhornOutput, OTTLRSinkhornOutput)):
            return float(self._output.reg_ot_cost)
        return float(self._output.reg_gw_cost)

    @property
    def converged(self) -> bool:  # noqa: D102
        return bool(self._output.converged)

    @property
    def potentials(self) -> Optional[Tuple[ArrayLike, ArrayLike]]:  # noqa: D102
        if isinstance(self._output, OTTSinkhornOutput):
            return self._output.f, self._output.g
        return None

    @property
    def rank(self) -> int:  # noqa: D102
        lin_output = self._output if self.is_linear else self._output.linear_state
        return len(lin_output.g) if isinstance(lin_output, OTTLRSinkhornOutput) else -1

    def _ones(self, n: int) -> ArrayLike:  # noqa: D102
        return jnp.ones((n,))  # type: ignore[return-value]
