from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

from anndata import AnnData

from moscot import _constants
from moscot._logging import logger
from moscot._types import ArrayLike, Str_Dict_t
from moscot.base.problems._mixins import AnalysisMixin, AnalysisMixinProtocol
from moscot.base.problems._utils import (
    _validate_annotations,
    _validate_args_cell_transition,
)
from moscot.base.problems.compound_problem import ApplyOutput_t, B, K
from moscot.plotting._utils import set_plotting_vars

__all__ = ["GenericAnalysisMixin"]


class GenericAnalysisMixinProtocol(AnalysisMixinProtocol[K, B], Protocol[K, B]):
    """Protocol class."""

    _batch_key: Optional[str]
    batch_key: Optional[str]
    adata: AnnData

    def _cell_transition(
        self: AnalysisMixinProtocol[K, B],
        *args: Any,
        **kwargs: Any,
    ) -> pd.DataFrame: ...


class GenericAnalysisMixin(AnalysisMixin[K, B]):
    """Generic Analysis Mixin."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._batch_key: Optional[str] = None

    def cell_transition(
        self: GenericAnalysisMixinProtocol[K, B],
        source: K,
        target: K,
        source_groups: Optional[Str_Dict_t] = None,
        target_groups: Optional[Str_Dict_t] = None,
        aggregation_mode: Literal["annotation", "cell"] = "annotation",
        forward: bool = False,
        batch_size: Optional[int] = None,
        normalize: bool = True,
        key_added: Optional[str] = _constants.CELL_TRANSITION,
    ) -> pd.DataFrame:
        """Aggregate the transport matrix.

        .. seealso::
            - See :doc:`../notebooks/examples/plotting/200_cell_transitions`
              on how to compute and plot the cell transitions.

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
            assert isinstance(self.batch_key, str)
        # TODO(michalk8): modify the behavior to match with the docs
        return self._cell_transition(
            key=self.batch_key,
            source=source,
            target=target,
            source_groups=source_groups,
            target_groups=target_groups,
            forward=forward,
            aggregation_mode=aggregation_mode,
            batch_size=batch_size,
            normalize=normalize,
            other_key=None,
            key_added=key_added,
        )

    def push(
        self: GenericAnalysisMixinProtocol[K, B],
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
            Source key in `solutions`.
        target
            Target key in `solutions`.
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
            if TYPE_CHECKING:
                assert isinstance(key_added, str)
            plot_vars = {
                "source": source,
                "target": target,
                "key": self.batch_key,
                "data": data if isinstance(data, str) else None,
                "subset": subset,
            }
            self.adata.obs[key_added] = self._flatten(result, key=self.batch_key)
            set_plotting_vars(self.adata, _constants.PUSH, key=key_added, value=plot_vars)
            return None
        return result

    def pull(
        self: GenericAnalysisMixinProtocol[K, B],
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
            Source key in `solutions`.
        target
            Target key in `solutions`.
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
                "key": self.batch_key,
                "data": data if isinstance(data, str) else None,
                "subset": subset,
                "source": source,
                "target": target,
            }
            self.adata.obs[key_added] = self._flatten(result, key=self.batch_key)
            set_plotting_vars(self.adata, _constants.PULL, key=key_added, value=plot_vars)
            return None
        return result

    @property
    def batch_key(self: GenericAnalysisMixinProtocol[K, B]) -> Optional[str]:
        """Batch key in :attr:`~anndata.AnnData.obs`."""
        return self._batch_key

    @batch_key.setter
    def batch_key(self: GenericAnalysisMixinProtocol[K, B], key: Optional[str]) -> None:
        if key is not None and key not in self.adata.obs:
            raise KeyError(f"Unable to find batch data in `adata.obs[{key!r}]`.")
        self._batch_key = key


class NeuralAnalysisMixin(AnalysisMixin[K, B]):
    """Analysis Mixin for Neural OT problems."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def cell_transition(
        self,
        source: Union[K, Tuple[K, AnnData, Union[str, Dict[str, str]]]],
        target: Union[K, Tuple[K, AnnData, Union[str, Dict[str, str]]]],
        source_groups: Str_Dict_t,
        target_groups: Str_Dict_t,
        forward: bool = False,  # return value will be row-stochastic if forward=True, else column-stochastic
        aggregation_mode: Literal["annotation", "cell"] = "cell",
        batch_size: int = 1024,
        normalize: bool = True,
        k: int = 30,
        key_added: Optional[str] = _constants.CELL_TRANSITION,
    ) -> Optional[pd.DataFrame]:
        """Compute a grouped transition matrix based on a pseudo-transport matrix.

        This function requires a projection of the velocity field onto existing cells, see
        :meth:`moscot.backends.ott.NeuralOutput.project_transport_matrix`.
        Afterwards, this function computes a transition matrix with entries corresponding to categories, e.g. cell
        types. The transition matrix will be row-stochastic if `forward` is `True`, otherwise column-stochastic.

        Parameters
        ----------
        source
            If of type `K`, key identifying the source distribution.
            If of type :class:`tuple``, the first argument is the key of the source distribution the model was
            trained on, the second argument of :class:`anndata.AnnData`, and the third element one of
                - `str`, then it must refer to a key in :attr:`anndata.AnnData.obsm`.
                - `dict`, then the dictionary stores `attr` (attribute of :class:`anndata.AnnData`) and `key`
                  (key of :class:`anndata.AnnData` ``['{attr}']``).
        target
            If of type `K`, key identifying the target distribution.
            If of type :class:`tuple``, the first argument is the key of the target distribution the model was
            trained on, the second argument of :class:`anndata.AnnData`, and the third element one of
                - `str`, then it must refer to a key in :attr:`anndata.AnnData.obsm`.
                - `dict`, then the dictionary stores `attr` (attribute of :class:`anndata.AnnData`) and `key`
                  (key of :class:`anndata.AnnData` ``['{attr}']``).
        source_groups
            Can be one of the following:
                - if `source_groups` is of type :class:`str` this should correspond to a key in
                  :attr:`anndata.AnnData.obs`. In this case, the categories in the transition matrix correspond to the
                  unique values in :attr:`anndata.AnnData.obs` ``['{source_groups}']``.
                - if `target_groups` is of type :class:`dict`, its key should correspond to a key in
                  :attr:`anndata.AnnData.obs` and its value to a list containing a subset of categories present in
                  :attr:`anndata.AnnData.obs` ``['{source_groups.keys()[0]}']``. The order of the list determines the
                  order in the transition matrix.
        target_groups
            Can be one of the following:
                - if `target_groups` is of type :class:`str` this should correspond to a key in
                  :attr:`anndata.AnnData.obs`. In this case, the categories in the transition matrix correspond to the
                  unique values in :attr:`anndata.AnnData.obs` ``['{target_groups}']``.
                - if `target_groups` is of :class:`dict`, its key should correspond to a key in
                  :attr:`anndata.AnnData.obs` and its value to a list containing a subset of categories present in
                  :attr:`anndata.AnnData.obs` ``['{target_groups.keys()[0]}']``. The order of the list determines the
                  order in the transition matrix.
        forward
            Computes transition from `source_annotations` to `target_annotations` if `True`, otherwise backward.
        aggregation_mode
            - `group`: transition probabilities from the groups defined by `source_annotation` are returned.
            - `cell`: the transition probabilities for each cell are returned.
        batch_size
            Number of data points the matrix-vector products are applied to at the same time.
            The larger, the more memory is required.
        normalize
            If `True` the transition matrix is normalized such that it is stochastic. If `forward` is `True`,
            the transition matrix is row-stochastic, otherwise column-stochastic.
        k
            Number of neighbors used to compute the pseudo-transport matrix if it hasn't been computed by
            :meth:`moscot.backends.ott.output.NeuralSolverOutput`
        key_added
            Key in :attr:`anndata.AnnData.uns` and/or :attr:`anndata.AnnData.obs` where the results
            for the corresponding plotting functions are stored.
            See TODO Notebook for how :mod:`moscot.plotting` works.

        Returns
        -------
        Aggregated transition matrix of cells or groups of cells.

        Notes
        -----
        To visualise the results, see :func:`moscot.pl.cell_transition`.
        """
        if isinstance(source, tuple):
            if len(source) != 2:
                raise ValueError("If `source` is a tuple it must be of length 2.")
            if not isinstance(source[0], AnnData):
                raise TypeError("The first element of the tuple must be of type AnnData.")
            if isinstance(source[1], str):
                source_data = source[0].obsm[source[1]]
            elif isinstance(source[1], dict):
                attr, val = next(iter(source[1]))
                source_data = getattr(source[0], attr)[val]
            else:
                raise TypeError("The second element of the tuple must be of type `str` or `dict`.")
            key_source, adata_src = source[1], source[0]
        else:
            key_source, source_data, adata_src = source, None, self.adata  # type:ignore[attr-defined]

        if isinstance(target, tuple):
            if len(target) != 2:
                raise ValueError("If `source` is a tuple it must be of length 2.")
            if not isinstance(target[0], AnnData):
                raise TypeError("The first element of the tuple must be of type AnnData.")
            if isinstance(target[1], str):
                target_data = target[0].obsm[target[1]]
            elif isinstance(target[1], dict):
                attr, val = next(iter(target[1]))
                target_data = getattr(target[0], attr)[val]
            else:
                raise TypeError("The second element of the tuple must be of type `str` or `dict`.")
            adata_tgt, key_target = target[0], target[1]
        else:
            key_target, target_data, adata_tgt = target, None, self.adata  # type:ignore[attr-defined]

        problem = self.problems[key_source, key_target]  # type:ignore[attr-defined]
        try:
            tm_result = problem.solution.transport_matrix if forward else problem.solution.inverse_transport_matrix
        except ValueError:
            logger.info(f"Projecting transport matrix based on {k} nearest neighbors.")
            tm_result = problem.project_transport_matrix(
                source_data, target_data, forward=forward, save_transport_matrix=True, batch_size=batch_size, k=k
            )

        annotation_key_source, annotations_present_source, annotations_ordered_source = _validate_args_cell_transition(
            adata_src, source_groups
        )
        annotation_key_target, annotations_present_target, annotations_ordered_target = _validate_args_cell_transition(
            adata_src, target_groups
        )
        df_source = (
            adata_src[adata_src.obs[self.temporal_key] == source]  # type:ignore[attr-defined]
            .obs[[annotation_key_source]]
            .copy()
        )
        df_target = (
            adata_tgt[adata_tgt.obs[self.temporal_key] == target]  # type:ignore[attr-defined]
            .obs[[annotation_key_target]]
            .copy()
        )
        annotations_verified_source, annotations_verified_target = _validate_annotations(
            df_source=df_source,
            df_target=df_target,
            source_annotation_key=annotation_key_source,
            target_annotation_key=annotation_key_target,
            source_annotations=annotations_present_source,
            target_annotations=annotations_present_target,
            aggregation_mode="annotation",
            forward=forward,
        )
        tm = pd.DataFrame(
            np.zeros((len(annotations_verified_source), len(annotations_verified_target))),
            index=annotations_verified_source,
            columns=annotations_verified_target,
        )
        for annotation_src in annotations_verified_source:
            for annotation_tgt in annotations_verified_target:
                tm.loc[annotation_src, annotation_tgt] = tm_result[
                    np.ix_((df_source == annotation_src).squeeze(), (df_target == annotation_tgt).squeeze())
                ].sum()
        annotations_ordered_source = tm.index if annotations_ordered_source is None else annotations_ordered_source
        annotations_ordered_target = tm.columns if annotations_ordered_target is None else annotations_ordered_target
        tm = tm.reindex(annotations_ordered_source)[annotations_ordered_target]
        if normalize:
            tm = tm.div(tm.sum(axis=int(forward)), axis=int(not forward))
        if key_added is not None:
            if aggregation_mode == "cell" and "cell" in self.adata.obs:  # type:ignore[attr-defined]
                raise KeyError(f"Aggregation is already present in `adata.obs[{aggregation_mode!r}]`.")
            plot_vars = {
                "transition_matrix": tm,
                "source": source,
                "target": target,
                "source_groups": source_groups,
                "target_groups": target_groups,
            }
            set_plotting_vars(
                self.adata,  # type:ignore[attr-defined]
                _constants.CELL_TRANSITION,
                key=key_added,
                value=plot_vars,
            )
        return tm

    def push(
        self,
        source: K,
        target: K,
        data: Optional[Union[str, ArrayLike]] = None,
        subset: Optional[Union[str, List[str], Tuple[int, int]]] = None,
        scale_by_marginals: bool = True,
        key_added: Optional[str] = _constants.PUSH,
        return_all: bool = False,
        return_data: bool = False,
        **kwargs: Any,
    ) -> Optional[ApplyOutput_t[K]]:
        """
        Push cells.

        Parameters
        ----------
        %(source)s
        %(target)s
        %(data)s
        %(subset)s
        %(scale_by_marginals)s
        %(key_added_plotting)s
        %(return_all)s
        %(return_data)s
        %(new_adata)s
        %(new_adata_joint_attr)s
        Return
        ------
        %(return_push_pull)s.
        """
        result = self._apply(  # type:ignore[attr-defined]
            start=source,
            end=target,
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
                "temporal_key": self.temporal_key,  # type:ignore[attr-defined]
            }
            self.adata.obs[key_added] = self._flatten(result, key=self.temporal_key)  # type:ignore[attr-defined, misc]
            set_plotting_vars(self.adata, _constants.PUSH, key=key_added, value=plot_vars)  # type:ignore[attr-defined]
        return result if return_data else None

    def pull(
        self,
        source: K,
        target: K,
        data: Optional[Union[str, ArrayLike]] = None,
        subset: Optional[Union[str, List[str], Tuple[int, int]]] = None,
        scale_by_marginals: bool = True,
        key_added: Optional[str] = _constants.PULL,
        return_all: bool = False,
        return_data: bool = False,
        **kwargs: Any,
    ) -> Optional[ApplyOutput_t[K]]:
        """
        Pull cells.

        Parameters
        ----------
        %(source)s
        %(target)s
        %(data)s
        %(subset)s
        %(scale_by_marginals)s
        %(key_added_plotting)s
        %(return_all)s
        %(return_data)s
        %(new_adata)s
        %(new_adata_joint_attr)s
        Return
        ------
        %(return_push_pull)s.
        """
        result = self._apply(  # type:ignore[attr-defined]
            start=source,
            end=target,
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
                "temporal_key": self.temporal_key,  # type:ignore[attr-defined]
            }
            self.adata.obs[key_added] = self._flatten(result, key=self.temporal_key)  # type:ignore[misc, attr-defined]
            set_plotting_vars(
                self.adata, _constants.PULL, key=key_added, value=plot_vars  # type:ignore[attr-defined]
            )
        return result if return_data else None
