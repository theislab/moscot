from typing import Any, Callable, List, Optional, Tuple, Union

import jaxlib.xla_extension as xla_ext

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from ott.neural.methods.flows.genot import GENOT
from ott.solvers.linear import sinkhorn, sinkhorn_lr
from ott.solvers.quadratic import gromov_wasserstein, gromov_wasserstein_lr

import matplotlib as mpl
import matplotlib.pyplot as plt

from moscot._types import ArrayLike, Device_t
from moscot.backends.ott._utils import get_nearest_neighbors
from moscot.base.output import BaseDiscreteSolverOutput, BaseNeuralOutput

__all__ = ["OTTOutput", "GraphOTTOutput", "OTTNeuralOutput"]


class OTTOutput(BaseDiscreteSolverOutput):
    """Output of various :term:`OT` problems.

    Parameters
    ----------
    output
        Output of the :mod:`ott` backend.
    """

    _NOT_COMPUTED = -1.0  # sentinel value used in `ott`

    def __init__(
        self,
        output: Union[
            sinkhorn.SinkhornOutput,
            sinkhorn_lr.LRSinkhornOutput,
            gromov_wasserstein.GWOutput,
            gromov_wasserstein_lr.LRGWOutput,
        ],
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
        return self._output.apply(
            x.T,
            axis=1 - forward,
        ).T  # convert to batch first

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
        output = self._output.linear_state if isinstance(self._output, gromov_wasserstein.GWOutput) else self._output
        return (
            len(output.g)
            if isinstance(output, (sinkhorn_lr.LRSinkhornOutput, gromov_wasserstein_lr.LRGWOutput))
            else -1
        )

    def _ones(self, n: int) -> ArrayLike:  # noqa: D102
        return jnp.ones((n,))


class OTTNeuralOutput(BaseNeuralOutput):
    """Output wrapper for GENOT."""

    def __init__(self, model: GENOT, logs: dict[str, list[float]]):
        """Initialize `OTTNeuralOutput`.

        Parameters
        ----------
        model : GENOT
            The OTT-Jax GENOT model
        """
        self._logs = logs
        self._model = model

    @property
    def logs(self):
        """Logs of the training.  A dictionary containing what the numeric values are i.e., loss.

        Returns
        -------
        dict[str, list[float]]
        """
        return self._logs

    def _project_transport_matrix(
        self,
        src_dist: ArrayLike,
        tgt_dist: ArrayLike,
        forward: bool,
        func: Callable[[jnp.ndarray], jnp.ndarray],
        save_transport_matrix: bool = False,  # TODO(@MUCDK) adapt order of arguments
        batch_size: int = 1024,
        k: int = 30,
        length_scale: Optional[float] = None,
        seed: int = 42,
        recall_target: float = 0.95,
        aggregate_to_topk: bool = True,
    ) -> sp.csr_matrix:
        row_indices: Union[jnp.ndarray, List[jnp.ndarray]] = []
        column_indices: Union[jnp.ndarray, List[jnp.ndarray]] = []
        distances_list: Union[jnp.ndarray, List[jnp.ndarray]] = []
        if length_scale is None:
            key = jax.random.PRNGKey(seed)
            src_batch = src_dist[jax.random.choice(key, src_dist.shape[0], shape=((batch_size,)))]
            tgt_batch = tgt_dist[jax.random.choice(key, tgt_dist.shape[0], shape=((batch_size,)))]
            length_scale = jnp.std(jnp.concatenate((func(src_batch), tgt_batch)))
        for index in range(0, len(src_dist), batch_size):
            distances, indices = get_nearest_neighbors(
                func(src_dist[index : index + batch_size, :]),
                tgt_dist,
                k,
                recall_target=recall_target,
                aggregate_to_topk=aggregate_to_topk,
            )
            distances = jnp.exp(-((distances / length_scale) ** 2))
            distances /= jnp.expand_dims(jnp.sum(distances, axis=1), axis=1)
            distances_list.append(distances.flatten())
            column_indices.append(indices.flatten())
            row_indices.append(
                jnp.repeat(jnp.arange(index, index + min(batch_size, len(src_dist) - index)), min(k, len(tgt_dist)))
            )
        distances = jnp.concatenate(distances_list)
        row_indices = jnp.concatenate(row_indices)
        column_indices = jnp.concatenate(column_indices)
        tm = sp.csr_matrix((distances, (row_indices, column_indices)), shape=[len(src_dist), len(tgt_dist)])
        if forward:
            if save_transport_matrix:
                self._transport_matrix = tm
        else:
            tm = tm.T
            if save_transport_matrix:
                self._inverse_transport_matrix = tm
        return tm

    def project_to_transport_matrix(  # type:ignore[override]
        self,
        src_cells: ArrayLike,
        tgt_cells: ArrayLike,
        forward: bool = True,
        condition: ArrayLike = None,
        save_transport_matrix: bool = False,  # TODO(@MUCDK) adapt order of arguments
        batch_size: int = 1024,
        k: int = 30,
        length_scale: Optional[float] = None,
        seed: int = 42,
        recall_target: float = 0.95,
        aggregate_to_topk: bool = True,
    ) -> sp.csr_matrix:
        """Project conditional neural OT map onto cells.

        In constrast to discrete OT, (conditional) neural OT does not necessarily map cells onto cells,
        but a cell can also be mapped to a location between two cells. This function computes
        a pseudo-transport matrix considering the neighborhood of where a cell is mapped to.
        Therefore, a neighborhood graph of `k` target cells is computed around each transported cell
        of the source distribution. The assignment likelihood of each mapped cell to the target cells is then
        computed with a Gaussian kernel with parameter `length_scale`.

        Parameters
        ----------
        condition
            Condition `src_cells` correspond to.
        src_cells
            Cells which are to be mapped.
        tgt_cells
            Cells from which the neighborhood graph around the mapped `src_cells` are computed.
        forward
            Whether to map cells based on the forward transport map or backward transport map.
        save_transport_matrix
            Whether to save the transport matrix.
        batch_size
            Number of data points in the source distribution the neighborhoodgraph is computed
            for in parallel.
        k
            Number of neighbors to construct the k-nearest neighbor graph of a mapped cell.
        length_scale
            Length scale of the Gaussian kernel used to compute the assignment likelihood. If `None`,
            `length_scale` is set to the empirical standard deviation of `batch_size` pairs of data points of the
            mapped source and target distribution.
        seed
            Random seed for sampling the pairs of distributions for computing the variance in case `length_scale`
            is `None`.
        recall_target
            Recall target for the approximation.
        aggregate_to_topk
            When true, the nearest neighbor aggregates approximate results to the top-k in sorted order.
            When false, returns the approximate results unsorted.
            In this case, the number of the approximate results is implementation defined and is greater or
            equal to the specified k.

        Returns
        -------
        The projected transport matrix.
        """
        src_cells, tgt_cells = jnp.asarray(src_cells), jnp.asarray(tgt_cells)
        push = self.push if condition is None else lambda x: self.push(x, condition)
        pull = self.pull if condition is None else lambda x: self.pull(x, condition)
        func, src_dist, tgt_dist = (push, src_cells, tgt_cells) if forward else (pull, tgt_cells, src_cells)
        return self._project_transport_matrix(
            src_dist=src_dist,
            tgt_dist=tgt_dist,
            forward=forward,
            func=func,
            save_transport_matrix=save_transport_matrix,  # TODO(@MUCDK) adapt order of arguments
            batch_size=batch_size,
            k=k,
            length_scale=length_scale,
            seed=seed,
            recall_target=recall_target,
            aggregate_to_topk=aggregate_to_topk,
        )

    def push(self, x: ArrayLike, cond: Optional[ArrayLike] = None) -> ArrayLike:
        """Push distribution `x` conditioned on condition `cond`.

        Parameters
        ----------
        x
            Distribution to push.
        cond
            Condition of conditional neural OT.

        Returns
        -------
        Pushed distribution.
        """
        if x.ndim not in (1, 2):
            raise ValueError(f"Expected 1D or 2D array, found `{x.ndim}`.")
        return self._apply(x, cond=cond, forward=True)

    def pull(self, x: ArrayLike, cond: Optional[ArrayLike] = None) -> ArrayLike:
        """Pull distribution `x` conditioned on condition `cond`.

        This does not make sense for some neural models and is therefore left unimplemented.

        Parameters
        ----------
         x
            Distribution to push.
        cond
            Condition of conditional neural OT.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("`pull` does not make sense for neural OT.")

    def _apply(self, x: ArrayLike, forward: bool, cond: Optional[ArrayLike] = None) -> ArrayLike:
        if not forward:
            raise NotImplementedError("Backward i.e., pull on neural OT is not supported.")
        return self._model.transport(x, condition=cond)

    @property
    def is_linear(self) -> bool:  # noqa: D102
        return True  # TODO(ilan-gold): need to contribute something to ott-jax so this is resolvable from GENOT

    @property
    def shape(self) -> Tuple[int, int]:
        """%(shape)s."""
        raise NotImplementedError()

    def to(
        self,
        device: Optional[Device_t] = None,
    ) -> "OTTNeuralOutput":
        """Transfer the output to another device or change its data type.

        Parameters
        ----------
        device
            If not `None`, the output will be transferred to `device`.

        Returns
        -------
        The output on a saved on `device`.
        """
        # # TODO(michalk8): when polishing docs, move the definition to the base class + use docrep
        # if isinstance(device, str) and ":" in device:
        #     device, ix = device.split(":")
        #     idx = int(ix)
        # else:
        #     idx = 0

        # if not isinstance(device, xla_ext.Device):
        #     try:
        #         device = jax.devices(device)[idx]
        #     except IndexError as err:
        #         raise IndexError(f"Unable to fetch the device with `id={idx}`.") from err

        # out = jax.device_put(self._model, device)
        # return OTTNeuralOutput(out)
        return self  # TODO(ilan-gold) move model to device

    @property
    def converged(self) -> bool:
        """%(converged)s."""
        # always return True for now
        return True


class GraphOTTOutput(OTTOutput):
    """Output of :term:`OT` problems with a graph geometry in the linear term.

    Parameters
    ----------
    output
        Output of the :mod:`ott` backend.
    shape
        Shape of the problem.
    """

    def __init__(
        self,
        output: Union[
            sinkhorn.SinkhornOutput,
            sinkhorn_lr.LRSinkhornOutput,
            gromov_wasserstein.GWOutput,
            gromov_wasserstein_lr.LRGWOutput,
        ],
        shape: Tuple[int, int],
    ):
        super().__init__(output)
        self._shape = shape

    @property
    def shape(self) -> Tuple[int, int]:  # noqa: D102
        return self._shape

    def _expand_data(self, x: jnp.ndarray, forward: bool) -> jnp.ndarray:
        if forward:
            shape = (self.shape[1],) if x.ndim == 1 else (self.shape[1], x.shape[1])
            return jnp.concatenate((x, jnp.zeros(shape)))
        shape = (self.shape[0],) if x.ndim == 1 else (self.shape[0], x.shape[1])
        return jnp.concatenate((jnp.zeros(shape), x))

    def _apply(self, x: ArrayLike, *, forward: bool) -> ArrayLike:
        x_expanded = self._expand_data(x, forward=forward)
        # ott-jax only supports lse_mode=False with graph geometry
        res = self._output.apply(x_expanded.T, axis=1 - forward, lse_mode=False).T
        return res[len(x) :] if forward else res[: -len(x)]

    def to(self, device: Optional[Device_t] = None) -> "GraphOTTOutput":  # noqa: D102
        if device is None:
            return GraphOTTOutput(jax.device_put(self._output, device=device), shape=self.shape)

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

        return GraphOTTOutput(jax.device_put(self._output, device), shape=self.shape)
