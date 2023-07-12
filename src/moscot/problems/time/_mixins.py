import itertools
import pathlib
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype, is_categorical_dtype, is_numeric_dtype

from anndata import AnnData

from moscot import _constants
from moscot._types import ArrayLike, Str_Dict_t
from moscot.base.problems._mixins import AnalysisMixin, AnalysisMixinProtocol
from moscot.base.problems.birth_death import BirthDeathProblem
from moscot.base.problems.compound_problem import ApplyOutput_t, B, K
from moscot.plotting._utils import set_plotting_vars
from moscot.utils.tagged_array import Tag

__all__ = ["TemporalMixin"]


class TemporalMixinProtocol(AnalysisMixinProtocol[K, B], Protocol[K, B]):  # type: ignore[misc]
    adata: AnnData
    problems: Dict[Tuple[K, K], BirthDeathProblem]
    temporal_key: Optional[str]
    _temporal_key: Optional[str]

    def cell_transition(  # noqa: D102
        self: "TemporalMixinProtocol[K, B]",
        source: K,
        target: K,
        source_groups: Str_Dict_t,
        target_groups: Str_Dict_t,
        forward: bool = False,  # return value will be row-stochastic if forward=True, else column-stochastic
        aggregation_mode: Literal["annotation", "cell"] = "annotation",
        batch_size: Optional[int] = None,
        normalize: bool = True,
        key_added: Optional[str] = _constants.CELL_TRANSITION,
    ) -> pd.DataFrame:
        ...

    def push(self, *args: Any, **kwargs: Any) -> Optional[ApplyOutput_t[K]]:
        ...

    def pull(self, *args: Any, **kwargs: Any) -> Optional[ApplyOutput_t[K]]:
        ...

    def _cell_transition(
        self: AnalysisMixinProtocol[K, B],
        *args: Any,
        **kwargs: Any,
    ) -> pd.DataFrame:
        ...

    def _sample_from_tmap(
        self: "TemporalMixinProtocol[K, B]",
        source: K,
        target: K,
        n_samples: int,
        source_dim: int,
        target_dim: int,
        batch_size: int = 256,
        account_for_unbalancedness: bool = False,
        interpolation_parameter: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tuple[List[Any], List[ArrayLike]]:
        ...

    def _compute_wasserstein_distance(
        self: "TemporalMixinProtocol[K, B]",
        point_cloud_1: ArrayLike,
        point_cloud_2: ArrayLike,
        a: Optional[ArrayLike] = None,
        b: Optional[ArrayLike] = None,
        backend: Literal["ott"] = "ott",
        **kwargs: Any,
    ) -> float:
        ...

    def _interpolate_gex_with_ot(
        self: "TemporalMixinProtocol[K, B]",
        number_cells: int,
        source_data: ArrayLike,
        target_data: ArrayLike,
        source: K,
        target: K,
        interpolation_parameter: float,
        account_for_unbalancedness: bool = True,
        batch_size: int = 256,
        seed: Optional[int] = None,
    ) -> ArrayLike:
        ...

    def _get_data(
        self: "TemporalMixinProtocol[K, B]",
        source: K,
        intermediate: Optional[K] = None,
        target: Optional[K] = None,
        posterior_marginals: bool = True,
        *,
        only_start: bool = False,
    ) -> Union[Tuple[ArrayLike, AnnData], Tuple[ArrayLike, ArrayLike, ArrayLike, AnnData, ArrayLike]]:
        ...

    def _interpolate_gex_randomly(
        self: "TemporalMixinProtocol[K, B]",
        number_cells: int,
        source_data: ArrayLike,
        target_data: ArrayLike,
        interpolation_parameter: float,
        growth_rates: Optional[ArrayLike] = None,
        seed: Optional[int] = None,
    ) -> ArrayLike:
        ...

    def _plot_temporal(
        self: "TemporalMixinProtocol[K, B]",
        data: Dict[K, ArrayLike],
        source: K,
        target: K,
        time_points: Optional[Iterable[K]] = None,
        basis: str = "umap",
        result_key: Optional[str] = None,
        fill_value: float = 0.0,
        save: Optional[Union[str, pathlib.Path]] = None,
        **kwargs: Any,
    ) -> None:
        ...

    @staticmethod
    def _get_interp_param(
        source: K, intermediate: K, target: K, interpolation_parameter: Optional[float] = None
    ) -> float:
        ...

    def __iter__(self) -> Iterator[Tuple[K, K]]:
        ...


class TemporalMixin(AnalysisMixin[K, B]):
    """Analysis Mixin for all problems involving a temporal dimension."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._temporal_key: Optional[str] = None

    def cell_transition(
        self: TemporalMixinProtocol[K, B],
        source: K,
        target: K,
        source_groups: Str_Dict_t,
        target_groups: Str_Dict_t,
        forward: bool = False,
        aggregation_mode: Literal["annotation", "cell"] = "annotation",
        batch_size: Optional[int] = None,
        normalize: bool = True,
        key_added: Optional[str] = _constants.CELL_TRANSITION,
    ) -> Optional[pd.DataFrame]:
        """Aggregate the transport matrix.

        .. seealso::
            - See :doc:`../notebooks/examples/plotting/200_cell_transitions` on how to
              compute and :func:`plot <moscot.plotting.cell_transition>` the cell transitions.

        Parameters
        ----------
        source
            Key identifying the source distribution.
        target
            Key identifying the target distribution.
        source_groups
            Source groups used for aggregation. Valid options are:

            - :class:`str` - key in :attr:`~anndata.AnnData.obs` where categorical data is stored.
            - :class:`dict` - a dictionary with one key corresponding to a categorical column in
              :attr:`~anndata.AnnData.obs` and values to a subset of categories.
        target_groups
            Target groups used for aggregation. Valid options are:

            - :class:`str` - key in :attr:`~anndata.AnnData.obs` where categorical data is stored.
            - :class:`dict` - a dictionary with one key corresponding to a categorical column in
              :attr:`~anndata.AnnData.obs` and values to a subset of categories.
        aggregation_mode
            How to aggregate the cell-level transport maps. Valid options are:

            - ``'annotation'`` - group the transitions by the ``source_groups`` and the ``target_groups``.
            - ``'cell'`` - do not group by the ``source_groups`` or the ``target_groups``, depending on the ``forward``.
        forward
            If :obj:`True`, compute the transitions from the ``source_groups`` to the ``target_groups``.
        batch_size
            Number of rows/columns of the cost matrix to materialize during :meth:`push` or :meth:`pull`.
            Larger value will require more memory.
        normalize
            If :obj:`True`, normalize the transition matrix. If ``forward = True``, the transition matrix
            will be row-stochastic, otherwise column-stochastic.
        key_added
            Key in :attr:`~anndata.AnnData.uns` where to save the result.

        Returns
        -------
        Depending on the ``key_added``:

        - :obj:`None` - returns the transition matrix.
        - :obj:`str` - returns nothing and saves the transition matrix to
          :attr:`uns['moscot_results']['cell_transition']['{key_added}'] <anndata.AnnData.uns>`
        """
        if TYPE_CHECKING:
            assert isinstance(self.temporal_key, str)
        return self._cell_transition(
            key=self.temporal_key,
            source=source,
            target=target,
            source_groups=source_groups,
            target_groups=target_groups,
            forward=forward,
            aggregation_mode=aggregation_mode,
            size=batch_size,
            normalize=normalize,
            key_added=key_added,
        )

    def sankey(
        self: "TemporalMixinProtocol[K, B]",
        source: K,
        target: K,
        source_groups: Str_Dict_t,
        target_groups: Str_Dict_t,
        threshold: Optional[float] = None,
        normalize: bool = False,
        forward: bool = True,
        restrict_to_existing: bool = True,
        order_annotations: Optional[Sequence[str]] = None,
        key_added: Optional[str] = _constants.SANKEY,
        **kwargs: Any,
    ) -> Optional[List[pd.DataFrame]]:
        """Compute a `Sankey diagram <https://en.wikipedia.org/wiki/Sankey_diagram>`_ between cells across time points.

        .. seealso::
            - See :doc:`../notebooks/examples/plotting/300_sankey` on how to
              compute and :func:`plot <moscot.plotting.sankey>` the Sankey diagram.

        Parameters
        ----------
        source
            Key identifying the source distribution.
        target
            Key identifying the target distribution.
        source_groups
            Source groups used for aggregation. Valid options are:

            - :class:`str` - key in :attr:`~anndata.AnnData.obs` where categorical data is stored.
            - :class:`dict` - a dictionary with one key corresponding to a categorical column in
              :attr:`~anndata.AnnData.obs` and values to a subset of categories.
        target_groups
            Target groups used for aggregation. Valid options are:

            - :class:`str` - key in :attr:`~anndata.AnnData.obs` where categorical data is stored.
            - :class:`dict` - a dictionary with one key corresponding to a categorical column in
              :attr:`~anndata.AnnData.obs` and values to a subset of categories.
        threshold
            Set cell transitions lower than ``threshold`` to :math:`0`.
        normalize
            If :obj:`True`, normalize the transition matrix. If ``forward = True``, the transition matrix
            will be row-stochastic, otherwise column-stochastic.
        forward
            If :obj:`True`, compute the transitions from the ``source_groups`` to the ``target_groups``.
        restrict_to_existing
            TODO(MUCDK)
        order_annotations
            Order of annotations from top to bottom. If :obj:`None`, use the order defined by the categories.
        key_added
            Key in :attr:`~anndata.AnnData.uns` where to save the result.
        kwargs
            Keyword arguments for :meth:`cell_transition`.

        Returns
        -------
        Depending on the ``key_added``:

        - :obj:`None` - returns the cell transitions.
        - :obj:`str` - returns nothing and saves the data for the diagram to
          :attr:`uns['moscot_results']['sankey']['{key_added}'] <anndata.AnnData.uns>`
        """
        tuples = self._policy.plan(start=source, end=target)
        cell_transitions = []
        for src, tgt in tuples:
            cell_transitions.append(
                self.cell_transition(
                    src,
                    tgt,
                    source_groups=source_groups,
                    target_groups=target_groups,
                    forward=forward,
                    normalize=normalize,
                    key_added=None,
                    **kwargs,
                )
            )

        if len(cell_transitions) > 1 and restrict_to_existing:
            for i in range(len(cell_transitions[:-1])):
                present_annotations = list(
                    set(cell_transitions[i + 1].index).intersection(set(cell_transitions[i].columns))
                )
                cell_transitions[i] = cell_transitions[i][present_annotations]

        if order_annotations is not None:
            cell_transitions_updated = []
            for ct in cell_transitions:
                order_annotations_present_index = [ann for ann in order_annotations if ann in ct.index]
                ct = ct.loc[order_annotations_present_index[::-1]]
                order_annotations_present_columns = [ann for ann in order_annotations if ann in ct.columns]
                ct = ct[order_annotations_present_columns[::-1]]
                cell_transitions_updated.append(ct)
        else:
            cell_transitions_updated = cell_transitions

        if threshold is not None:
            for ct in cell_transitions:
                ct[ct < threshold] = 0.0

        if key_added is None:
            return cell_transitions_updated

        if isinstance(source_groups, str):
            key = source_groups
        # TODO(MUCDK): should this be really `target_groups`?
        elif isinstance(target_groups, dict):
            key = list(target_groups.keys())[0]
        else:
            raise TypeError(f"Expected early cells to be either `str` or `dict`, found `{type(source_groups)}`.")

        plot_vars = {
            "transition_matrices": cell_transitions_updated,
            "key": key,
            "source": source,
            "target": target,
            "source_groups": source_groups,
            "target_groups": target_groups,
            "captions": [str(t) for t in tuples],
        }
        set_plotting_vars(self.adata, _constants.SANKEY, key=key_added, value=plot_vars)  # noqa: RET503

    def push(
        self: TemporalMixinProtocol[K, B],
        source: K,
        target: K,
        data: Optional[Union[str, ArrayLike]] = None,
        subset: Optional[Union[str, List[str], Tuple[int, int]]] = None,
        scale_by_marginals: bool = True,
        key_added: Optional[str] = _constants.PUSH,
        return_all: bool = False,
        **kwargs: Any,
    ) -> Optional[ApplyOutput_t[K]]:
        """Push mass from source to target.

        Parameters
        ----------
        source
            Source key in :attr:`solutions`.
        target
            Target key in :attr:`solutions`.
        data
            Initial data to push, see :meth:`~moscot.base.problems.OTProblem.push` for information.
        subset
            Push values contained only within the subset.
        scale_by_marginals
            Whether to scale by the source :term:`marginals`.
        key_added
            Key in :attr:`~anndata.AnnData.obs` where to add the result.
        return_all
            Whether to also return intermediate results. Always true if ``key_added != None``.
        kwargs
            Keyword arguments for the subproblems' :meth:`~moscot.base.problems.OTProblem.push` method.

        Returns
        -------
        Depending on the ``key_added``:

        - :obj:`None` - returns the result.
        - :class:`str` - returns nothing and updates :attr:`obs['{key_added}'] <anndata.AnnData.obs>`
          with the result.
        """
        result = self._apply(
            source=source,
            target=target,
            data=data,
            subset=subset,
            forward=True,
            return_all=return_all or key_added is not None,
            scale_by_marginals=scale_by_marginals,
            **kwargs,
        )

        if TYPE_CHECKING:
            assert isinstance(result, dict)

        if key_added is not None:
            plot_vars = {
                "source": source,
                "target": target,
                "temporal_key": self.temporal_key,
                "data": data if isinstance(data, str) else None,
                "subset": subset,
            }
            self.adata.obs[key_added] = self._flatten(result, key=self.temporal_key)
            set_plotting_vars(self.adata, _constants.PUSH, key=key_added, value=plot_vars)
            return None
        return result

    def pull(
        self: TemporalMixinProtocol[K, B],
        source: K,
        target: K,
        data: Optional[Union[str, ArrayLike]] = None,
        subset: Optional[Union[str, List[str], Tuple[int, int]]] = None,
        scale_by_marginals: bool = True,
        key_added: Optional[str] = _constants.PULL,
        return_all: bool = False,
        **kwargs: Any,
    ) -> Optional[ApplyOutput_t[K]]:
        """Pull mass from target to source.

        Parameters
        ----------
        source
            Source key in :attr:`solutions`.
        target
            Target key in :attr:`solutions`.
        data
            Initial data to pull, see :meth:`~moscot.base.problems.OTProblem.pull` for information.
        subset
            Pull values contained only within the subset.
        scale_by_marginals
            Whether to scale by the source :term:`marginals`.
        key_added
            Key in :attr:`~anndata.AnnData.obs` where to add the result.
        return_all
            Whether to also return intermediate results. Always true if ``key_added != None``.
        kwargs
            Keyword arguments for the subproblems' :meth:`~moscot.base.problems.OTProblem.pull` method.

        Returns
        -------
        Depending on the ``key_added``:

        - :obj:`None` - returns the result.
        - :class:`str` - returns nothing and updates :attr:`obs['{key_added}'] <anndata.AnnData.obs>`
          with the result.
        """
        result = self._apply(
            source=source,
            target=target,
            data=data,
            subset=subset,
            forward=False,
            return_all=return_all or key_added is not None,
            scale_by_marginals=scale_by_marginals,
            **kwargs,
        )
        if TYPE_CHECKING:
            assert isinstance(result, dict)

        if key_added is not None:
            plot_vars = {
                "temporal_key": self.temporal_key,
                "data": data if isinstance(data, str) else None,
                "subset": subset,
                "source": source,
                "target": target,
            }
            self.adata.obs[key_added] = self._flatten(result, key=self.temporal_key)
            set_plotting_vars(self.adata, _constants.PULL, key=key_added, value=plot_vars)
            return None
        return result

    @property
    def prior_growth_rates(self: TemporalMixinProtocol[K, B]) -> Optional[pd.DataFrame]:
        """Prior estimate of the source growth rates."""
        computed = [isinstance(p.prior_growth_rates, np.ndarray) for p in self.problems.values()]
        if not np.sum(computed):
            return None

        cols = ["prior_growth_rates"]
        df_list = [
            pd.DataFrame(problem.prior_growth_rates, index=problem.adata.obs_names, columns=cols)
            for problem in self.problems.values()
        ]
        df_1 = pd.concat(df_list, verify_integrity=True)
        indices_remaining = set(self.adata.obs_names) - set(df_1.index)
        df_2 = pd.DataFrame(np.nan, index=list(indices_remaining), columns=cols)

        return pd.concat([df_1, df_2], verify_integrity=True)

    @property
    def posterior_growth_rates(self: TemporalMixinProtocol[K, B]) -> Optional[pd.DataFrame]:
        """Posterior estimate of the source growth rates."""
        computed = [isinstance(p.posterior_growth_rates, np.ndarray) for p in self.problems.values()]
        if not np.sum(computed):
            return None

        cols = ["posterior_growth_rates"]
        df_list = [
            pd.DataFrame(problem.posterior_growth_rates, index=problem.adata.obs_names, columns=cols)
            for problem in self.problems.values()
        ]
        df_1 = pd.concat(df_list, verify_integrity=True)
        indices_remaining = set(self.adata.obs_names) - set(df_1.index)
        df_2 = pd.DataFrame(np.nan, index=list(indices_remaining), columns=cols)

        return pd.concat([df_1, df_2], verify_integrity=True)

    @property
    def cell_costs_source(self: TemporalMixinProtocol[K, B]) -> Optional[pd.DataFrame]:
        """Cell cost obtained by the :term:`first dual potential <dual potentials>`.

        Only available for subproblems with :attr:`problem_kind = 'linear' <problem_kind>`.
        """
        computed = [isinstance(s.potentials, tuple) for s in self.solutions.values()]
        if not np.sum(computed):
            return None

        cols = ["cell_cost_source"]
        # TODO(michalk8): `[1]` will fail if potentials is None
        df_list = [
            pd.DataFrame(
                np.asarray(problem.solution.potentials[0]),  # type: ignore[union-attr,index]
                index=problem.adata_src.obs_names,
                columns=cols,
            )
            for problem in self.problems.values()
        ]
        df_1 = pd.concat(df_list, verify_integrity=True)
        indices_remaining = set(self.adata.obs_names) - set(df_1.index)
        df_2 = pd.DataFrame(np.nan, index=list(indices_remaining), columns=cols)
        return pd.concat([df_1, df_2], verify_integrity=True)

    @property
    def cell_costs_target(self: TemporalMixinProtocol[K, B]) -> Optional[pd.DataFrame]:
        """Cell cost obtained by the :term:`second dual potential <dual potentials>`.

        Only available for subproblems with :attr:`problem_kind = 'linear' <problem_kind>`.
        """
        computed = [isinstance(s.potentials, tuple) for s in self.solutions.values()]
        if not np.sum(computed):
            return None

        cols = ["cell_cost_target"]
        # TODO(michalk8): `[1]` will fail if potentials is None
        df_list = [
            pd.DataFrame(
                np.array(problem.solution.potentials[1]),  # type: ignore[union-attr,index]
                index=problem.adata_tgt.obs_names,
                columns=cols,
            )
            for problem in self.problems.values()
        ]
        df_1 = pd.concat(df_list, verify_integrity=True)
        indices_remaining = set(self.adata.obs_names) - set(df_1.index)
        df_2 = pd.DataFrame(np.nan, index=list(indices_remaining), columns=cols)
        return pd.concat([df_1, df_2], verify_integrity=True)

    def _get_data(
        self: TemporalMixinProtocol[K, B],
        source: K,
        intermediate: Optional[K] = None,
        target: Optional[K] = None,
        posterior_marginals: bool = True,
        *,
        only_start: bool = False,
    ) -> Union[Tuple[ArrayLike, AnnData], Tuple[ArrayLike, ArrayLike, ArrayLike, AnnData, ArrayLike]]:
        # TODO: use .items()
        for src, tgt in self.problems:
            tag = self.problems[src, tgt].xy.tag  # type: ignore[union-attr]
            if tag != Tag.POINT_CLOUD:
                raise ValueError(
                    f"Expected `tag={Tag.POINT_CLOUD}`, "  # type: ignore[union-attr]
                    f"found `tag={self.problems[src, tgt].xy.tag}`."
                )
            if src == source:
                source_data = self.problems[src, tgt].xy.data_src  # type: ignore[union-attr]
                if only_start:
                    return source_data, self.problems[src, tgt].adata_src
                # TODO(michalk8): posterior marginals
                attr = "posterior_growth_rates" if posterior_marginals else "prior_growth_rates"
                growth_rates_source = getattr(self.problems[src, tgt], attr)
                break
        else:
            raise ValueError(f"No data found for `{source}` time point.")
        for src, tgt in self.problems:
            if src == intermediate:
                intermediate_data = self.problems[src, tgt].xy.data_src  # type: ignore[union-attr]
                intermediate_adata = self.problems[src, tgt].adata_src
                break
        else:
            raise ValueError(f"No data found for `{intermediate}` time point.")
        for src, tgt in self.problems:
            if tgt == target:
                target_data = self.problems[src, tgt].xy.data_tgt  # type: ignore[union-attr]
                break
        else:
            raise ValueError(f"No data found for `{target}` time point.")

        return (  # type:ignore[return-value]
            source_data,
            growth_rates_source,
            intermediate_data,
            intermediate_adata,
            target_data,
        )

    def compute_interpolated_distance(
        self: TemporalMixinProtocol[K, B],
        source: K,
        intermediate: K,
        target: K,
        interpolation_parameter: Optional[float] = None,
        n_interpolated_cells: Optional[int] = None,
        account_for_unbalancedness: bool = False,
        batch_size: int = 256,
        posterior_marginals: bool = True,
        seed: Optional[int] = None,
        backend: Literal["ott"] = "ott",
        **kwargs: Any,
    ) -> float:
        """Compute `Wasserstein distance <https://en.wikipedia.org/wiki/Wasserstein_metric>`_ between
        :term:`OT`-interpolated and intermediate cells.

        .. seealso::
            - TODO(MUCDK): create an example showing the usage.

        This is a validation method which interpolates cells between the ``source`` and ``target`` distributions
        leveraging the :term:`OT` coupling to approximate cells at the ``intermediate`` time point.

        Parameters
        ----------
        source
            Key identifying the source distribution.
        intermediate
            Key identifying the intermediate distribution.
        target
            Key identifying the target distribution.
        interpolation_parameter
            Interpolation parameter in :math:`(0, 1)` defining the weight of the ``source`` and ``target``
            distributions. If :obj:`None`, it is linearly interpolated.
        n_interpolated_cells
            Number of cells used for interpolation. If :obj:`None`, use the number of cells in the ``intermediate``
            distribution.
        account_for_unbalancedness
            Whether to account for unbalancedness by assuming exponential cell growth and death.
        batch_size
            Number of rows/columns of the cost matrix to materialize during :meth:`push` or :meth:`pull`.
            Larger value will require more memory.
        posterior_marginals
            Whether to use :attr:`posterior_growth_rates` or :attr:`prior_growth_rates`.
            TODO(MUCDK): needs more explanation
        seed
            Random seed used when sampling the interpolated cells.
        backend
            Backend used for the distance computation.
        kwargs
            Keyword arguments for the distance function, depending on the ``backend``:

            - ``'ott'`` - :func:`~ott.tools.sinkhorn_divergence.sinkhorn_divergence`.

        Returns
        -------
        The distance between :term:`OT`-interpolated cells and cells at the ``intermediate`` time point.
        It is recommended to compare this to the distances computed by :meth:`compute_time_point_distances` and
        :meth:`compute_random_distance`.
        """  # noqa: D205
        source_data, _, intermediate_data, _, target_data = self._get_data(  # type: ignore[misc]
            source,
            intermediate,
            target,
            posterior_marginals=posterior_marginals,
            only_start=False,
        )
        interpolation_parameter = self._get_interp_param(
            source, intermediate, target, interpolation_parameter=interpolation_parameter
        )
        n_interpolated_cells = n_interpolated_cells if n_interpolated_cells is not None else len(intermediate_data)
        interpolation = self._interpolate_gex_with_ot(
            number_cells=n_interpolated_cells,
            source_data=source_data,
            target_data=target_data,
            source=source,
            target=target,
            interpolation_parameter=interpolation_parameter,
            account_for_unbalancedness=account_for_unbalancedness,
            batch_size=batch_size,
            seed=seed,
        )
        return self._compute_wasserstein_distance(intermediate_data, interpolation, backend=backend, **kwargs)

    def compute_random_distance(
        self: TemporalMixinProtocol[K, B],
        source: K,
        intermediate: K,
        target: K,
        interpolation_parameter: Optional[float] = None,
        n_interpolated_cells: Optional[int] = None,
        account_for_unbalancedness: bool = False,
        posterior_marginals: bool = True,
        seed: Optional[int] = None,
        backend: Literal["ott"] = "ott",
        **kwargs: Any,
    ) -> float:
        """Compute `Wasserstein distance <https://en.wikipedia.org/wiki/Wasserstein_metric>`_ between randomly
        interpolated and intermediate cells.

        .. seealso::
            - TODO(MUCDK): create an example showing the usage.

        This function interpolates cells between the ``source`` and ``target`` distributions using a random
        :term:`OT` coupling to approximate cells at the ``intermediate`` time point.

        Parameters
        ----------
        source
            Key identifying the source distribution.
        intermediate
            Key identifying the intermediate distribution.
        target
            Key identifying the target distribution.
        interpolation_parameter
            Interpolation parameter in :math:`(0, 1)` defining the weight of the ``source`` and ``target``
            distributions. If :obj:`None`, it is linearly interpolated.
        n_interpolated_cells
            Number of cells used for interpolation. If :obj:`None`, use the number of cells in the ``intermediate``
            distribution.
        account_for_unbalancedness
            Whether to account for unbalancedness by assuming exponential cell growth and death.
        posterior_marginals
            Whether to use :attr:`posterior_growth_rates` or :attr:`prior_growth_rates`.
            TODO(MUCDK): needs more explanation
        seed
            Random seed used when sampling the interpolated cells.
        backend
            Backend used for the distance computation.
        kwargs
            Keyword arguments for the distance function, depending on the ``backend``:

            - ``'ott'`` - :func:`~ott.tools.sinkhorn_divergence.sinkhorn_divergence`.

        Returns
        -------
        The distance between randomly interpolated cells and cells at the ``intermediate`` time point.
        """  # noqa: D205
        source_data, growth_rates_source, intermediate_data, _, target_data = self._get_data(  # type: ignore[misc]
            source, intermediate, target, posterior_marginals=posterior_marginals, only_start=False
        )

        interpolation_parameter = self._get_interp_param(
            source, intermediate, target, interpolation_parameter=interpolation_parameter
        )
        n_interpolated_cells = n_interpolated_cells if n_interpolated_cells is not None else len(intermediate_data)

        growth_rates = growth_rates_source if account_for_unbalancedness else None
        random_interpolation = self._interpolate_gex_randomly(
            n_interpolated_cells,
            source_data,
            target_data,
            interpolation_parameter=interpolation_parameter,
            growth_rates=growth_rates,
            seed=seed,
        )
        return self._compute_wasserstein_distance(intermediate_data, random_interpolation, backend=backend, **kwargs)

    def compute_time_point_distances(
        self: TemporalMixinProtocol[K, B],
        source: K,
        intermediate: K,
        target: K,
        posterior_marginals: bool = True,
        backend: Literal["ott"] = "ott",
        **kwargs: Any,
    ) -> Tuple[float, float]:
        """Compute `Wasserstein distance <https://en.wikipedia.org/wiki/Wasserstein_metric>`_ between time points.

        .. seealso::
            - TODO(MUCDK): create an example showing the usage.

        Parameters
        ----------
        source
            Key identifying the source distribution.
        intermediate
            Key identifying the intermediate distribution.
        target
            Key identifying the target distribution.
        posterior_marginals
            Whether to use :attr:`posterior_growth_rates` or :attr:`prior_growth_rates`.
            TODO(MUCDK): needs more explanation
        backend
            Backend used for the distance computation.
        kwargs
            Keyword arguments for the distance function, depending on the ``backend``:

            - ``'ott'`` - :func:`~ott.tools.sinkhorn_divergence.sinkhorn_divergence`.

        Returns
        -------
        The distance between ``source`` and ``intermediate`` cells and
        ``intermediate`` and ``target`` cells, respectively.
        """
        source_data, _, intermediate_data, _, target_data = self._get_data(  # type: ignore[misc]
            source,
            intermediate,
            target,
            posterior_marginals=posterior_marginals,
            only_start=False,
        )
        distance_source_intermediate = self._compute_wasserstein_distance(
            point_cloud_1=source_data, point_cloud_2=intermediate_data, backend=backend, **kwargs
        )
        distance_intermediate_target = self._compute_wasserstein_distance(
            point_cloud_1=intermediate_data, point_cloud_2=target_data, backend=backend, **kwargs
        )

        return distance_source_intermediate, distance_intermediate_target

    def compute_batch_distances(
        self: TemporalMixinProtocol[K, B],
        time: K,
        batch_key: str,
        posterior_marginals: bool = True,
        backend: Literal["ott"] = "ott",
        **kwargs: Any,
    ) -> float:
        """Compute the average `Wasserstein distance <https://en.wikipedia.org/wiki/Wasserstein_metric>`_ between
        batches for a specific time point.

        .. seealso::
            - TODO(MUCDK): create an example showing the usage.

        Parameters
        ----------
        time
            Time point for which to compute the distances.
        batch_key
            Key in :attr:`~anndata.AnnData.obs` where batches are stored.
        posterior_marginals
            Whether to use :attr:`posterior_growth_rates` or :attr:`prior_growth_rates`.
            TODO(MUCDK): needs more explanation
        backend
            Backend used for the distance computation.
        kwargs
            Keyword arguments for the distance function, depending on the ``backend``:

            - ``'ott'`` - :func:`~ott.tools.sinkhorn_divergence.sinkhorn_divergence`.

        Returns
        -------
        The average distance between batches for a specific time point.
        """  # noqa: D205
        data, adata = self._get_data(time, posterior_marginals=posterior_marginals, only_start=True)  # type: ignore[misc] # noqa: E501
        if len(data) != len(adata):
            raise ValueError(f"Expected the `data` to have length `{len(adata)}`, found `{len(data)}`.")

        dist: List[float] = []
        for batch_1, batch_2 in itertools.combinations(adata.obs[batch_key].unique(), 2):
            dist.append(
                self._compute_wasserstein_distance(
                    point_cloud_1=data[(adata.obs[batch_key] == batch_1).values],
                    point_cloud_2=data[(adata.obs[batch_key] == batch_2).values],
                    backend=backend,
                    **kwargs,
                )
            )
        return np.mean(dist)  # type: ignore[return-value]

    # TODO(@MUCDK) possibly offer two alternatives, once exact EMD with POT backend and once approximate,
    # faster with same solver as used for original problems
    def _compute_wasserstein_distance(
        self: TemporalMixinProtocol[K, B],
        point_cloud_1: ArrayLike,
        point_cloud_2: ArrayLike,
        a: Optional[ArrayLike] = None,
        b: Optional[ArrayLike] = None,
        backend: Literal["ott"] = "ott",
        **kwargs: Any,
    ) -> float:
        if backend == "ott":
            from moscot.backends.ott import sinkhorn_divergence

            return sinkhorn_divergence(point_cloud_1, point_cloud_2, a, b, **kwargs)
        raise NotImplementedError("Only `ott` available as backend.")

    def _interpolate_gex_with_ot(
        self: TemporalMixinProtocol[K, B],
        number_cells: int,
        source_data: ArrayLike,
        target_data: ArrayLike,
        source: K,
        target: K,
        interpolation_parameter: float,
        account_for_unbalancedness: bool = True,
        batch_size: int = 256,
        seed: Optional[int] = None,
    ) -> ArrayLike:
        rows_sampled, cols_sampled = self._sample_from_tmap(
            source=source,
            target=target,
            n_samples=number_cells,
            source_dim=len(source_data),
            target_dim=len(target_data),
            batch_size=batch_size,
            account_for_unbalancedness=account_for_unbalancedness,
            interpolation_parameter=interpolation_parameter,
            seed=seed,
        )
        return (
            source_data[np.repeat(rows_sampled, [len(col) for col in cols_sampled]), :] * (1 - interpolation_parameter)
            + target_data[np.hstack(cols_sampled), :] * interpolation_parameter
        )

    def _interpolate_gex_randomly(
        self: TemporalMixinProtocol[K, B],
        number_cells: int,
        source_data: ArrayLike,
        target_data: ArrayLike,
        interpolation_parameter: float,
        growth_rates: Optional[ArrayLike] = None,
        seed: Optional[int] = None,
    ) -> ArrayLike:
        rng = np.random.RandomState(seed)
        if growth_rates is None:
            row_probability = np.ones(len(source_data))
        else:
            row_probability = growth_rates ** (1 - interpolation_parameter)
        row_probability /= np.sum(row_probability)
        return (
            source_data[rng.choice(len(source_data), size=number_cells, p=row_probability), :]
            * (1 - interpolation_parameter)
            + target_data[rng.choice(len(target_data), size=number_cells), :] * interpolation_parameter
        )

    @staticmethod
    def _get_interp_param(
        source: K, intermediate: K, target: K, interpolation_parameter: Optional[float] = None
    ) -> float:
        if TYPE_CHECKING:
            assert isinstance(source, float)
            assert isinstance(intermediate, float)
            assert isinstance(target, float)
        if interpolation_parameter is not None:
            if 0 < interpolation_parameter < 1:
                return interpolation_parameter
            raise ValueError(
                f"Expected interpolation parameter to be in interval `(0, 1)`, found `{interpolation_parameter}`."
            )

        if source < intermediate < target:
            return (intermediate - source) / (target - source)
        raise ValueError(
            f"Expected intermediate time point to be in interval `({source}, {target})`, found `{intermediate}`."
        )

    @property
    def temporal_key(self) -> Optional[str]:
        """Temporal key in :attr:`~anndata.AnnData.obs`."""
        return self._temporal_key

    @temporal_key.setter
    def temporal_key(self: TemporalMixinProtocol[K, B], key: Optional[str]) -> None:
        if key is None:
            self._temporal_key = key
            return
        if key not in self.adata.obs:
            raise KeyError(f"Unable to find temporal key in `adata.obs[{key!r}]`.")
        col = self.adata.obs[key]
        if not (is_categorical_dtype(col) and is_numeric_dtype(col.cat.categories)):
            raise TypeError(
                f"Expected `adata.obs[{key!r}]` to be categorical with numeric categories, "
                f"found `{infer_dtype(col)}`."
            )
        self._temporal_key = key
