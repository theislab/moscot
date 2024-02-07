from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jaxlib.xla_extension as xla_ext

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from ott.problems.linear import potentials
from ott.solvers.linear import sinkhorn, sinkhorn_lr
from ott.solvers.quadratic import gromov_wasserstein, gromov_wasserstein_lr
from ott.neural.models.base_solver import BaseNeuralSolver # TODO(ilan-gold): package structure will change when michaln reviews

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from moscot._types import ArrayLike, Device_t
from moscot.backends.ott._utils import ConditionalDualPotentials, get_nearest_neighbors
from moscot.base.output import BaseDiscreteSolverOutput, BaseNeuralOutput

__all__ = ["OTTOutput", "NeuralDualOutput", "CondNeuralDualOutput", "ConditionalDualPotentials"]

Train_t = Dict[str, Union[float, List[float]]]


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
        output = self._output.linear_state if isinstance(self._output, gromov_wasserstein.GWOutput) else self._output
        return (
            len(output.g)
            if isinstance(output, (sinkhorn_lr.LRSinkhornOutput, gromov_wasserstein_lr.LRGWOutput))
            else -1
        )

    def _ones(self, n: int) -> ArrayLike:  # noqa: D102
        return jnp.ones((n,))


class OTTNeuralOutput(BaseNeuralOutput):
    """Base class for OTT neural OT output."""

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
        row_indices: Union[jnp.ndarray, List[jnp.ndarray]] = []
        column_indices: Union[jnp.ndarray, List[jnp.ndarray]] = []
        distances_list: Union[jnp.ndarray, List[jnp.ndarray]] = []
        if length_scale is None:
            key = jax.random.PRNGKey(seed)
            src_batch = src_dist[jax.random.choice(key, src_dist.shape[0], shape=((batch_size,)))]
            tgt_batch = tgt_dist[jax.random.choice(key, tgt_dist.shape[0], shape=((batch_size,)))]
            length_scale = jnp.std(jnp.concatenate((func(src_batch), tgt_batch)))
        for index in range(0, len(src_dist), batch_size):
            distances, indices = get_nearest_neighbors(func(src_dist[index : index + batch_size, :]), tgt_dist, k)
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


class NeuralDualOutput(OTTNeuralOutput):
    """
    Output representation of neural OT problems.

    Parameters
    ----------
    output
        The trained model as :class:`ott.problems.linear.potentials.DualPotentials`.
    training_logs
        Statistics of the model training.
    """

    def __init__(self, output: potentials.DualPotentials, model: BaseNeuralSolver, training_logs: Train_t):
        self._output = output
        self._model = model
        self._training_logs = training_logs
        self._transport_matrix: ArrayLike = None
        self._inverse_transport_matrix: ArrayLike = None

    def _apply(self, x: ArrayLike, *, forward: bool) -> ArrayLike:
        return self._output.transport(x, forward=forward)

    def plot_convergence(
        self,
        data: str = "loss",
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
            Which curve to plot.
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
        return super().plot_convergence(  # type: ignore[misc]
            data={data: self._training_logs[data]},
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
        """Return if problem is linear."""
        return True

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
    ) -> "NeuralDualOutput":
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
            except IndexError as err:
                raise IndexError(f"Unable to fetch the device with `id={idx}`.") from err

        out = jax.device_put(self._output, device)  # TODO: also move self._model to device
        return NeuralDualOutput(out, self._model, self.training_logs)

    @property
    def cost(self) -> float:
        """Predicted optimal transport cost on validation dataset."""
        return self._training_logs["predicted_cost"]  # type: ignore[return-value]

    @property
    def converged(self) -> bool:
        """%(converged)s."""
        # always return True for now
        return True

    @property
    def potentials(
        self,
    ) -> Tuple[Callable[[jnp.ndarray], float], Callable[[jnp.ndarray], float]]:
        """Return the learned potential functions."""
        f = jax.vmap(self._output.f)
        g = jax.vmap(self._output.g)
        return f, g

    def push(self, x: ArrayLike) -> ArrayLike:
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

    def pull(self, x: ArrayLike) -> ArrayLike:
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

    def evaluate_a(self, x: ArrayLike) -> ArrayLike:
        """Marginals of the source distribution."""
        if self._model.mlp_xi is None:
            raise ValueError("The source marginals have not been traced.")
        return self._model.state_eta.apply_fn({"params": self._model.state_eta.params}, x)  # type:ignore[union-attr]

    def evaluate_b(self, x: ArrayLike) -> ArrayLike:
        """Marginals of the target distribution."""
        if self._model.mlp_eta is None:
            raise ValueError("The target marginals have not been traced.")
        return self._model.state_xi.apply_fn({"params": self._model.state_xi.params}, x)  # type:ignore[union-attr]

    def _ones(self, n: int) -> jnp.ndarray:
        return jnp.ones((n,))

    def _format_params(self, fmt: Callable[[Any], str]) -> str:
        params = {
            "predicted_cost": round(self.cost, 3),
        }
        return ", ".join(f"{name}={fmt(val)}" for name, val in params.items())


class CondNeuralDualOutput(NeuralDualOutput):
    """
    Output representation of conditional neural OT problems.

    Parameters
    ----------
    output
        The trained model as :class:`moscot.backends.ott._utils.ConditionalDualPotentials`.
    training_logs
        Statistics of the model training.
    """

    def __init__(self, output: ConditionalDualPotentials, model: Any, **kwargs):  # TODO: adapt type
        super().__init__(output=output, model=model, **kwargs)
        self._output = output

    def _apply(self, cond: ArrayLike, x: ArrayLike, *, forward: bool) -> ArrayLike:  # type:ignore[override]
        cond = jnp.array(cond)
        return self._output.transport(cond, x, forward=forward)

    def project_transport_matrix(  # type:ignore[override]
        self,
        condition: ArrayLike,
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

        Returns
        -------
        The projected transport matrix.
        """
        src_cells, tgt_cells = jnp.asarray(src_cells), jnp.asarray(tgt_cells)
        func, src_dist, tgt_dist = (
            (partial(self.push, condition), src_cells, tgt_cells)
            if forward
            else (partial(self.pull, condition), tgt_cells, src_cells)
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
    ) -> "CondNeuralDualOutput":
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
            except IndexError as err:
                raise IndexError(f"Unable to fetch the device with `id={idx}`.") from err

        out = jax.device_put(self._output, device)
        # TODO: allow model to be transferred to device
        return CondNeuralDualOutput(output=out, model=self._model, training_logs=self.training_logs)

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

    def evaluate_a(self, cond: ArrayLike, x: ArrayLike) -> ArrayLike:
        """Conditional marginals of the source distribution."""
        if self._model.mlp_eta is None:
            raise ValueError("The source marginals have not been traced.")
        if cond.ndim != 2:
            cond = cond[:, None]
        input = jnp.concatenate((x, cond), axis=-1)
        return self._model.state_eta.apply_fn(  # type:ignore[union-attr]
            {"params": self._model.state_eta.params}, input  # type:ignore[union-attr]
        )

    def evaluate_b(self, cond: ArrayLike, x: ArrayLike) -> ArrayLike:
        """Conditional marginals of the target distribution."""
        if self._model.mlp_xi is None:
            raise ValueError("The target marginals have not been traced.")
        if cond.ndim != 2:
            cond = cond[:, None]
        input = jnp.concatenate((x, cond), axis=-1)
        return self._model.state_xi.apply_fn({"params": self._model.state_xi.params}, input)  # type:ignore[union-attr]
