from functools import partial
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import jaxlib.xla_extension as xla_ext
import numpy as np
import jax
import jax.numpy as jnp
import scipy.sparse as sp
from moscot._docs._docs import d
from ott.problems.linear.potentials import DualPotentials
from ott.solvers.linear.sinkhorn import SinkhornOutput as OTTSinkhornOutput
from ott.solvers.linear.sinkhorn_lr import LRSinkhornOutput as OTTLRSinkhornOutput
from ott.solvers.quadratic.gromov_wasserstein import GWOutput as OTTGWOutput

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure

from moscot._types import ArrayLike, Device_t
from moscot.backends.ott._utils import ConditionalDualPotentials, get_nearest_neighbors
from moscot.base.output import BaseNeuralOutput, BaseSolverOutput

__all__ = ["OTTOutput", "NeuralOutput", "ConditionalNeuralOutput"]

Train_t = Dict[str, Dict[str, List[float]]]


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
        data: Optional[Dict[str, List[float]]] = None,
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
        data
            Data containing information on convergence.
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

        def select_values(
            last_k: Optional[int] = None, data: Optional[Dict[str, List[float]]] = None
        ) -> Tuple[str, jnp.ndarray, jnp.ndarray]:
            if data is None:  # this is for discrete OT classes
                # `> 1` because of pure Sinkhorn
                if len(self._costs) > 1 or self._errors is None:
                    metric = self._costs
                    metric_str = "cost"
                else:
                    metric = self._errors
                    metric_str = "error"
            else:  # this is for Monge Maps
                if len(data) > 1:
                    raise ValueError(f"`data` must have length 1, but found {len(data)}.")
                metric = list(data.values())[0]
                metric_str = list(data.keys())[0]

            last_k = min(last_k, len(metric)) if last_k is not None else len(metric)
            return metric_str, metric[-last_k:], range(len(metric))[-last_k:]

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        kind, values, xs = select_values(last_k, data=data)

        ax.plot(xs, values, **kwargs)
        ax.set_xlabel("iteration (logged)")
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

    def _apply(self, x: ArrayLike, *, forward: bool) -> ArrayLike:  # type:ignore[override]
        if x.ndim == 1:
            return self._output.apply(x, axis=1 - forward)
        return self._output.apply(x.T, axis=1 - forward).T  # convert to batch first

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

    @property
    def shape(self) -> Tuple[int, int]:
        if isinstance(self._output, OTTSinkhornOutput):
            return self._output.f.shape[0], self._output.g.shape[0]
        return self._output.geom.shape

    @property
    def transport_matrix(self) -> ArrayLike:
        return self._output.matrix

    @property
    def is_linear(self) -> bool:  # noqa: D102
        return isinstance(self._output, (OTTSinkhornOutput, OTTLRSinkhornOutput))

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


class NeuralOutput(ConvergencePlotterMixin, BaseNeuralOutput):
    """
    Output representation of neural OT problems.

    Parameters
    ----------
    output
        The trained model as :class:`ott.problems.linear.potentials.DualPotentials`.
    training_logs
        Statistics of the model training.
    """

    def __init__(self, output: DualPotentials, training_logs: Train_t):
        self._output = output
        self._training_logs = training_logs
        self._transport_matrix: ArrayLike = None
        self._inverse_transport_matrix: ArrayLike = None

    def _apply(self, x: ArrayLike, *, forward: bool) -> ArrayLike:
        return self._output.transport(x, forward=forward)

    def plot_convergence(  # type: ignore[override]
        self,
        data: Dict[Literal["pretrain", "train", "valid"], str] = MappingProxyType({"train": "loss"}),
        last_k: Optional[int] = None,
        title: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        save: Optional[str] = None,
        return_fig: bool = False,
        **kwargs: Any,
    ) -> Optional[Figure]:
        """Plot the convergence curve.

        Parameters
        ----------
        data
            Which training curve to plot.
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
        """
        if len(data) > 1:
            raise ValueError(f"`data` must be of length 1, but found {len(data)}.")
        k, v = next(iter(data.items()))
        return super().plot_convergence(
            data={k + ": " + v: self._training_logs[f"{k}_logs"][v]},
            last_k=last_k,
            title=title,
            figsize=figsize,
            dpi=dpi,
            save=save,
            return_fig=return_fig,
            **kwargs,
        )

    @property
    def training_logs(self) -> Train_t:
        """Training logs."""
        return self._training_logs

    @property
    def shape(self) -> Tuple[int, int]:
        """%(shape)s."""
        raise NotImplementedError()

    def is_linear(self) -> bool:
        return True

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
    ) -> sp.csr_matrix:
        get_knn_fn = jax.vmap(get_nearest_neighbors, in_axes=(0, None, None))
        row_indices: Union[jnp.ndarray, List[jnp.ndarray]] = []
        column_indices: Union[jnp.ndarray, List[jnp.ndarray]] = []
        distances_list: Union[jnp.ndarray, List[jnp.ndarray]] = []
        if length_scale is None:
            key = jax.random.PRNGKey(seed)
            src_batch = src_dist[jax.random.choice(key, src_dist.shape[0], shape=((batch_size,)))]
            tgt_batch = tgt_dist[jax.random.choice(key, tgt_dist.shape[0], shape=((batch_size,)))]
            length_scale = jnp.std(jnp.concatenate((func(src_batch), tgt_batch)))
        for index in range(0, len(src_dist), batch_size):
            distances, indices = get_knn_fn(func(src_dist[index : index + batch_size]), tgt_dist, k)
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

    def project_transport_matrix(  # type:ignore[override]
        self,
        src_cells: ArrayLike,
        tgt_cells: ArrayLike,
        forward: bool = True,
        save_transport_matrix: bool = False,  # TODO(@MUCDK) adapt order of arguments
        batch_size: int = 1024,
        k: int = 30,
        length_scale: Optional[float] = None,
        seed: int = 42,
    ) -> sp.csr_matrix:
        """Project neural OT map onto cells.

        In constrast to discrete OT, Neural OT does not necessarily map cells onto cells,
        but a cell can also be mapped to a location between two cells. This function computes
        a pseudo-transport matrix considering the neighborhood of where a cell is mapped to.
        Therefore, a neighborhood graph of `k` target cells is computed around each transported cell
        of the source distribution. The assignment likelihood of each mapped cell to the target cells is then
        computed with a Gaussian kernel with parameter `length_scale`.

        Parameters
        ----------
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

        Returns
        -------
        The projected transport matrix.
        """
        src_cells, tgt_cells = jnp.asarray(src_cells), jnp.asarray(tgt_cells)
        func, src_dist, tgt_dist = (self.push, src_cells, tgt_cells) if forward else (self.pull, tgt_cells, src_cells)
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
        )

    @property
    def transport_matrix(self) -> ArrayLike:
        """Projected transport matrix."""
        if self._transport_matrix is None:
            raise ValueError(
                "The projected transport matrix has not been computed yet." " Please call `project_transport_matrix`."
            )
        return self._transport_matrix

    @property
    def inverse_transport_matrix(self) -> ArrayLike:
        """Projected transport matrix based on the inverse map."""
        if self._inverse_transport_matrix is None:
            raise ValueError(
                "The inverse projected transport matrix has not been computed yet."
                " Please call `project_transport_matrix`."
            )
        return self._inverse_transport_matrix

    def to(
        self,
        device: Optional[Device_t] = None,
    ) -> "NeuralOutput":
        """Transfer the output to another device or change its data type.

        Parameters
        ----------
        device
            If not `None`, the output will be transferred to `device`.

        Returns
        -------
        The output on a saved on `device`.
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
                raise IndexError(f"Unable to fetch the device with `id={idx}`.")

        out = jax.device_put(self._output, device)
        return NeuralOutput(out, self.training_logs)

    @property
    def cost(self) -> float:
        """Predicted optimal transport cost."""
        return self.training_logs["valid_logs"]["predicted_cost"]

    @property
    def converged(self) -> bool:
        """%(converged)s."""
        # always return True for now
        return True

    @property
    def potentials(  # type: ignore[override]
        self,
    ) -> Tuple[Callable[[jnp.ndarray], float], Callable[[jnp.ndarray], float]]:
        """Return the learned potential functions."""
        f = jax.vmap(self._output.f)
        g = jax.vmap(self._output.g)
        return f, g

    def push(self, x: ArrayLike) -> ArrayLike:  # type: ignore[override]
        """Push distribution `x`.

        Parameters
        ----------
        x
            Distribution to push.

        Returns
        -------
        Pushed distribution.
        """
        if x.ndim not in (1, 2):
            raise ValueError(f"Expected 1D or 2D array, found `{x.ndim}`.")
        return self._apply(x, forward=True)

    def pull(self, x: ArrayLike) -> ArrayLike:  # type: ignore[override]
        """Pull distribution `x`.

        Parameters
        ----------
        x
            Distribution to pull.

        Returns
        -------
        Pulled distribution.
        """
        if x.ndim not in (1, 2):
            raise ValueError(f"Expected 1D or 2D array, found `{x.ndim}`.")
        return self._apply(x, forward=False)

    def evaluate_f(self, x: ArrayLike) -> ArrayLike:
        """Apply forward potential to `x`.

        Parameters
        ----------
        x
            Distribution to apply potential to.

        Returns
        -------
        Forward potential evaluated at `x`.
        """
        if x.ndim not in (1, 2):
            raise ValueError(f"Expected 1D or 2D array, found `{x.ndim}`.")
        return jax.vmap(self._output.f)(x)

    def evaluate_g(self, x: ArrayLike) -> ArrayLike:
        """Apply backward potential to `x`.

        Parameters
        ----------
        x
            Distribution to apply backward potential to.

        Returns
        -------
        Backward potential evaluated at `x`.
        """
        if x.ndim not in (1, 2):
            raise ValueError(f"Expected 1D or 2D array, found `{x.ndim}`.")
        return jax.vmap(self._output.g)(x)

    @property
    def a(self) -> ArrayLike:
        """Marginals of the source distribution."""
        # TODO: adapt when tracing marginals
        raise NotImplementedError()

    @property
    def b(self) -> ArrayLike:
        """Marginals of the target distribution."""
        # TODO: adapt when tracing marginals
        raise NotImplementedError()

    def _ones(self, n: int) -> jnp.ndarray:
        return jnp.ones((n,))

    def _format_params(self, fmt: Callable[[Any], str]) -> str:
        if "sinkhorn_dist" in self.training_logs["valid_logs"]:
            params = {
                "predicted_cost": round(self.cost, 3),
                "best_loss": round(self.training_logs["valid_logs"]["best_loss"], 3),  # type: ignore[call-overload]
                "sinkhorn_dist": round(self.training_logs["valid_logs"]["sinkhorn_dist"], 3),  # type: ignore[call-overload]
            }
        else:
            params = {
                "predicted_cost": round(self.cost, 3),
            }
        return ", ".join(f"{name}={fmt(val)}" for name, val in params.items())


class ConditionalNeuralOutput(NeuralOutput):
    """
    Output representation of conditional neural OT problems.

    Parameters
    ----------
    output
        The trained model as :class:`moscot.backends.ott._utils.ConditionalDualPotentials`.
    training_logs
        Statistics of the model training.
    """

    def __init__(self, output: ConditionalDualPotentials, **kwargs):
        super().__init__(output=output, **kwargs)
        self._output = output

    def _apply(self, cond: ArrayLike, x: ArrayLike, *, forward: bool) -> ArrayLike:  # type:ignore[override]
        cond = jnp.array(cond)
        return self._output.transport(cond, x, forward=forward)

    def project_transport_matrix(  # type:ignore[override]
        self,
        cond: ArrayLike,
        src_cells: ArrayLike,
        tgt_cells: ArrayLike,
        forward: bool = True,
        save_transport_matrix: bool = False,  # TODO(@MUCDK) adapt order of arguments
        batch_size: int = 1024,
        k: int = 30,
        length_scale: Optional[float] = None,
        seed: int = 42,
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
        cond
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

        Returns
        -------
        The projected transport matrix.
        """
        src_cells, tgt_cells = jnp.asarray(src_cells), jnp.asarray(tgt_cells)
        func, src_dist, tgt_dist = (
            (partial(self.push, cond), src_cells, tgt_cells)
            if forward
            else (partial(self.pull, cond), tgt_cells, src_cells)
        )
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
        )

    @property
    def transport_matrix(self) -> ArrayLike:
        """Projected transport matrix."""
        if self._transport_matrix is None:
            raise ValueError(
                "The projected transport matrix has not been computed yet." " Please call `project_transport_matrix`."
            )
        return self._transport_matrix

    @property
    def inverse_transport_matrix(self) -> ArrayLike:
        """Projected transport matrix based on the inverse map."""
        if self._inverse_transport_matrix is None:
            raise ValueError(
                "The inverse projected transport matrix has not been computed yet."
                " Please call `project_transport_matrix`."
            )
        return self._inverse_transport_matrix

    def to(
        self,
        device: Optional[Device_t] = None,
    ) -> "ConditionalNeuralOutput":
        """Transfer the output to another device or change its data type.

        Parameters
        ----------
        device
            If not `None`, the output will be transferred to `device`.

        Returns
        -------
        The output on a saved on `device`.
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
                raise IndexError(f"Unable to fetch the device with `id={idx}`.")

        out = jax.device_put(self._output, device)
        return ConditionalNeuralOutput(output=out, training_logs=self.training_logs)

    def push(self, cond: ArrayLike, x: ArrayLike) -> ArrayLike:  # type: ignore[override]
        """Push distribution `x` conditioned on condition `cond`.

        Parameters
        ----------
        cond
            Condition of conditional neural OT.
        x
            Distribution to push.

        Returns
        -------
        Pushed distribution.
        """
        if x.ndim not in (1, 2):
            raise ValueError(f"Expected 1D or 2D array, found `{x.ndim}`.")
        return self._apply(cond, x, forward=True)

    def pull(self, cond: ArrayLike, x: ArrayLike) -> ArrayLike:  # type: ignore[override]
        """Pull distribution `x` conditioned on condition `cond`.

        Parameters
        ----------
        cond
            Condition of conditional neural OT.
        x
            Distribution to pull.

        Returns
        -------
        Pulled distribution.
        """
        if x.ndim not in (1, 2):
            raise ValueError(f"Expected 1D or 2D array, found `{x.ndim}`.")
        return self._apply(cond, x, forward=False)

    def evaluate_f(self, cond: ArrayLike, x: ArrayLike) -> ArrayLike:  # type:ignore[override]
        """Apply forward potential to `x` conditionend on condition `cond`.

        Parameters
        ----------
        cond
            Condition of conditional neural OT.
        x
            Distribution to apply potential to.

        Returns
        -------
        Forward potential evaluated at `x`.
        """
        if x.ndim not in (1, 2):
            raise ValueError(f"Expected 1D or 2D array, found `{x.ndim}`.")
        return jax.vmap(self._output.f)(cond, x)

    def evaluate_g(self, cond: ArrayLike, x: ArrayLike) -> ArrayLike:  # type:ignore[override]
        """Apply backward potential to `x` conditionend on condition `cond`.

        Parameters
        ----------
        cond
            Condition of conditional neural OT.
        x
            Distribution to apply backward potential to.

        Returns
        -------
        Backward potential evaluated at `x`.
        """
        if x.ndim not in (1, 2):
            raise ValueError(f"Expected 1D or 2D array, found `{x.ndim}`.")
        return jax.vmap(self._output.g)(cond, x)
