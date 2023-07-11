from typing import Any, Optional, Tuple, Union

import jaxlib.xla_extension as xla_ext

import jax
import jax.numpy as jnp
import numpy as np
from ott.solvers.linear import sinkhorn, sinkhorn_lr
from ott.solvers.quadratic import gromov_wasserstein

import matplotlib as mpl
import matplotlib.pyplot as plt

from moscot._types import ArrayLike, Device_t
from moscot.base.output import BaseSolverOutput

__all__ = ["OTTOutput"]


class OTTOutput(BaseSolverOutput):
    """Output of various :term:`OT` problems.

    Parameters
    ----------
    output
        Output of the :mod:`ott` backend.
    """

    _NOT_COMPUTED = -1.0  # sentinel value used in `ott`

    def __init__(
        self, output: Union[sinkhorn.SinkhornOutput, sinkhorn_lr.LRSinkhornOutput, gromov_wasserstein.GWOutput]
    ):
        super().__init__()
        self._output = output
        self._costs = None if isinstance(output, sinkhorn.SinkhornOutput) else output.costs
        self._errors = output.errors

    def plot_costs(
        self,
        last: Optional[int] = None,
        title: Optional[str] = None,
        return_fig: bool = False,
        ax: Optional[mpl.axes.Axes] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        save: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[mpl.figure.Figure]:
        """Plot regularized :term:`OT` costs during the iterations.

        Parameters
        ----------
        last
            How many of the last steps of the algorithm to plot. If :obj:`None`, plot the full curve.
        title
            Title of the plot. If :obj:`None`, it is determined automatically.
        return_fig
            Whether to return the figure.
        ax
            Axes on which to plot.
        figsize
            Size of the figure.
        dpi
            Dots per inch.
        save
            Path where to save the figure.
        kwargs
            Keyword arguments for :meth:`matplotlib.axes.Axes.plot`.

        Returns
        -------
        If ``return_fig = True``, return the figure.
        """
        if self._costs is None:
            raise RuntimeError("No costs to plot.")

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi) if ax is None else (ax.get_figure(), ax)
        self._plot_lines(ax, np.asarray(self._costs), last=last, y_label="cost", title=title, **kwargs)

        if save is not None:
            fig.savefig(save)
        return fig if return_fig else None

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
        """Plot errors along iterations.

        Parameters
        ----------
        last
            Number of errors corresponding at the ``last`` steps of the algorithm to plot. If :obj:`None`,
            plot the full curve.
        title
            Title of the plot. If :obj:`None`, it is determined automatically.
        outer_iteration
            Which outermost iteration's errors to plot.
            Only used when this is the solution to the :term:`quadratic problem`.
        return_fig
            Whether to return the figure.
        ax
            Axes on which to plot.
        figsize
            Size of the figure.
        dpi
            Dots per inch.
        save
            Path where to save the figure.
        kwargs
            Keyword arguments for :meth:`matplotlib.axes.Axes.plot`.

        Returns
        -------
        If ``return_fig = True``, return the figure.
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
        if isinstance(self._output, sinkhorn.SinkhornOutput):
            return self._output.f.shape[0], self._output.g.shape[0]
        return self._output.geom.shape

    @property
    def transport_matrix(self) -> ArrayLike:  # noqa: D102
        return self._output.matrix

    @property
    def is_linear(self) -> bool:  # noqa: D102
        return isinstance(self._output, (sinkhorn.SinkhornOutput, sinkhorn_lr.LRSinkhornOutput))

    def to(self, device: Optional[Device_t] = None) -> "OTTOutput":  # noqa: D102
        if device is None:
            return OTTOutput(jax.device_put(self._output, device=device))

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
        return float(self._output.reg_ot_cost if self.is_linear else self._output.reg_gw_cost)

    @property
    def converged(self) -> bool:  # noqa: D102
        return bool(self._output.converged)

    @property
    def potentials(self) -> Optional[Tuple[ArrayLike, ArrayLike]]:  # noqa: D102
        if isinstance(self._output, sinkhorn.SinkhornOutput):
            return self._output.f, self._output.g
        return None

    @property
    def rank(self) -> int:  # noqa: D102
        lin_output = self._output if self.is_linear else self._output.linear_state
        return len(lin_output.g) if isinstance(lin_output, sinkhorn_lr.LRSinkhornOutput) else -1

    def _ones(self, n: int) -> ArrayLike:  # noqa: D102
        return jnp.ones((n,))
