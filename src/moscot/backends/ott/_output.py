from types import MappingProxyType
from typing import Any, Dict, List, Tuple, Union, Literal, Callable, Optional

from matplotlib.figure import Figure
import scipy.sparse as sp
import matplotlib.pyplot as plt

from ott.solvers.linear.sinkhorn import SinkhornOutput as OTTSinkhornOutput
from ott.problems.linear.potentials import DualPotentials
from ott.solvers.linear.sinkhorn_lr import LRSinkhornOutput as OTTLRSinkhornOutput
from ott.solvers.quadratic.gromov_wasserstein import GWOutput as OTTGWOutput
import jax
import jax.numpy as jnp
import jaxlib.xla_extension as xla_ext

from moscot._types import Device_t, ArrayLike
from moscot.solvers._output import BaseSolverOutput
from moscot.backends.ott._utils import get_nearest_neighbors

__all__ = ["OTTOutput", "NeuralOutput"]

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


class NeuralOutput(ConvergencePlotterMixin, BaseSolverOutput):
    """
    Output representation of neural OT problems.
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
        data: Dict[
            Literal["pretrain", "train", "valid"], Literal["loss", "w_dist", "penalty", "loss_g", "loss_f"]
        ] = MappingProxyType({"train": "loss"}),
        last_k: Optional[int] = None,
        title: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        save: Optional[str] = None,
        return_fig: bool = False,
        **kwargs: Any,
    ) -> Optional[Figure]:
        if len(data) > 1:
            raise ValueError(f"`data` must be of length 1, but found {len(data)}.")
        k, v = next(iter(data.items()))
        return super().plot_convergence(
            data={k + ": " + v: self._training_logs[k][v]},
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
        """%(shape)s"""
        raise NotImplementedError()

    def project_transport_matrix(
        self,
        src: ArrayLike,
        tgt: ArrayLike,
        forward: bool = True,
        save_transport_matrix: bool = True,
        batch_size: int = 1024,
        k: int = 30,
        length_scale: Optional[float] = None,
        seed: int = 42,
    ) -> sp.csr_matrix:
        """Project Monge Map onto a transport matrix."""
        src, tgt = jnp.asarray(src), jnp.asarray(tgt)
        func = self.push if forward else self.pull
        src_dist, tgt_dist = src, tgt if forward else tgt, src
        get_knn_fn = jax.vmap(get_nearest_neighbors, in_axes=(0, None, None))
        row_indices: Union[jnp.ndarray, List[jnp.ndarray]] = []
        column_indices: Union[jnp.ndarray, List[jnp.ndarray]] = []
        distances_list: Union[jnp.ndarray, List[jnp.ndarray]] = []
        if length_scale is None:
            key = jax.random.PRNGKey(seed)
            src_batch = jax.random.choice(key, src.shape[0], shape=(batch_size))
            tgt_batch = jax.random.choice(key, tgt.shape[0], shape=(batch_size))
            length_scale = jnp.std(jnp.concatenate(src_batch, tgt_batch))
        for index in range(0, len(src_dist), batch_size):
            # compute k nearest neighbors for current source batch compared to whole target
            distances, indices = get_knn_fn(func(src_dist[index : index + batch_size]), tgt_dist, k)
            distances = jnp.exp(-((distances / length_scale) ** 2))
            distances /= jnp.expand_dims(jnp.sum(distances, axis=1), axis=1)
            distances_list.append(distances.flatten())
            column_indices.append(indices.flatten())
            row_indices.append(
                jnp.repeat(jnp.arange(index, index + min(batch_size, len(src_dist) - index)), min(k, len(tgt)))
            )
        # create sparse matrix with normalized exp(-d(x,y)) as entries
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

    @property
    def transport_matrix(self) -> ArrayLike:
        """%(transport_matrix)s"""
        if self._transport_matrix is None:
            raise ValueError(
                "The forward projected transport matrix has not been computed yet."
                " Please call `project_transport_matrix`."
            )
        return self._transport_matrix

    @property
    def inverse_transport_matrix(self) -> ArrayLike:
        """%(inverse_transport_matrix)s"""
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
        """
        Transfer the output to another device or change its data type.
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
        return self.training_logs["valid_logs"]["predicted_cost"][0]

    @property
    def converged(self) -> bool:
        """%(converged)s."""
        # always return True for now
        return True

    @property
    def potentials(  # type: ignore[override]
        self,
    ) -> Tuple[Callable[[jnp.ndarray], float], Callable[[jnp.ndarray], float]]:
        """Returns the two learned potential functions."""
        f = jax.vmap(self._output.f)
        g = jax.vmap(self._output.g)
        return f, g

    def push(self, x: ArrayLike) -> ArrayLike:  # type: ignore[override]
        if x.ndim not in (1, 2):
            raise ValueError(f"Expected 1D or 2D array, found `{x.ndim}`.")
        return self._apply(x, forward=True)

    def pull(self, x: ArrayLike) -> ArrayLike:  # type: ignore[override]
        if x.ndim not in (1, 2):
            raise ValueError(f"Expected 1D or 2D array, found `{x.ndim}`.")
        return self._apply(x, forward=False)

    def push_potential(self, x: ArrayLike) -> ArrayLike:
        if x.ndim not in (1, 2):
            raise ValueError(f"Expected 1D or 2D array, found `{x.ndim}`.")
        return jax.vmap(self._output.f)(x)

    def pull_potential(self, x: ArrayLike) -> ArrayLike:
        if x.ndim not in (1, 2):
            raise ValueError(f"Expected 1D or 2D array, found `{x.ndim}`.")
        return jax.vmap(self._output.g)(x)

    @property
    def a(self) -> ArrayLike:
        """
        Marginals of the source distribution.
        """
        # TODO: adapt when tracing marginals
        raise NotImplementedError()

    @property
    def b(self) -> ArrayLike:
        """
        Marginals of the target distribution.
        """
        # TODO: adapt when tracing marginals
        raise NotImplementedError()

    def _ones(self, n: int) -> jnp.ndarray:
        return jnp.ones((n,))

    def _format_params(self, fmt: Callable[[Any], str]) -> str:
        params = {
            "predicted_cost": round(self.cost, 3),
            "best_loss": round(self.training_logs["valid_logs"]["best_loss"][0], 3),
            "sink_dist": round(self.training_logs["valid_logs"]["sink_dist"][0], 3),
        }
        return ", ".join(f"{name}={fmt(val)}" for name, val in params.items())
