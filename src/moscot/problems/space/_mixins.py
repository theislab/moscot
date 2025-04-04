import itertools
import types
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as st
from scipy.linalg import svd
from scipy.spatial import ConvexHull
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from anndata import AnnData

from moscot import _constants
from moscot._logging import logger
from moscot._types import ArrayLike, Device_t, Str_Dict_t
from moscot.base.problems._mixins import AnalysisMixin
from moscot.base.problems.compound_problem import B, K
from moscot.base.problems.problem import AbstractSpSc
from moscot.utils.subset_policy import StarPolicy

__all__ = ["SpatialAlignmentMixin", "SpatialMappingMixin"]


class SpatialAlignmentMixin(AnalysisMixin[K, B]):
    """Spatial alignment mixin class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._spatial_key: Optional[str] = None
        self._batch_key: Optional[str] = None

    def _interpolate_scheme(
        self,
        reference: K,
        mode: Literal["warp", "affine"],
        spatial_key: str,
    ) -> Tuple[Dict[K, ArrayLike], Optional[Dict[K, Optional[ArrayLike]]]]:
        """Scheme for interpolation."""
        # get reference
        src = self._subset_spatial(reference, spatial_key=spatial_key)
        transport_maps: Dict[K, ArrayLike] = {reference: src}
        transport_metadata: Dict[K, Optional[ArrayLike]] = {}
        if mode == "affine":
            src -= src.mean(0)
            transport_metadata = {reference: np.diag((1, 1))}  # 2d data

        # get the reference
        reference_ = [reference] if isinstance(reference, str) else reference
        full_steps = self._policy._graph
        starts = set(itertools.chain.from_iterable(full_steps)) - set(reference_)  # type: ignore[arg-type]

        if mode == "affine":
            _transport = _affine
        elif mode == "warp":
            _transport = _warp
        else:
            raise NotImplementedError(f"Alignment mode `{mode!r}` is not yet implemented.")

        steps = {}
        for start in starts:
            try:
                steps[start, reference, True] = self._policy.plan(start=start, end=reference)
            except nx.NetworkXNoPath:
                steps[reference, start, False] = self._policy.plan(start=reference, end=start)

        for (start, end, forward), path in steps.items():
            tmap = self._interpolate_transport(path=path, scale_by_marginals=True)
            # make `tmap` to have shape `(m, n_ref)` and apply it to `src` of shape `(n_ref, d)`
            key, tmap = (start, tmap) if forward else (end, tmap.T)
            spatial_data = self._subset_spatial(key, spatial_key=spatial_key)
            transport_maps[key], transport_metadata[key] = _transport(tmap, src=src, tgt=spatial_data)

        # TODO(michalk8): always return the metadata?
        return transport_maps, (transport_metadata if mode == "affine" else None)

    def align(
        self,
        reference: Optional[K] = None,
        mode: Literal["warp", "affine"] = "warp",
        spatial_key: Optional[str] = None,
        key_added: Optional[str] = None,
    ) -> Optional[Tuple[ArrayLike, Optional[Dict[K, Optional[ArrayLike]]]]]:
        """Align the spatial data.

        Parameters
        ----------
        reference
            Reference key. If a :class:`star policy <moscot.utils.subset_policy.StarPolicy>` was used,
            its reference will always be used.
        mode
            Alignment mode. Valid options are:

            - ``'warp'`` - warp the data to the ``reference``.
            - ``'affine'`` - align the data to the ``reference`` using affine transformation.
        spatial_key
            Key in :attr:`~anndata.AnnData.obsm` where the spatial coordinates are stored.
            If :obj:`None`, use :attr:`spatial_key`.
        key_added
            Key in :attr:`~anndata.AnnData.obsm` and :attr:`~anndata.AnnData.uns` where to store the alignment.

        Returns
        -------
        Depending on the ``key_added``:

        - :obj:`None` - returns the aligned coordinates and metadata.
          The metadata is :obj:`None` when ``mode != 'affine'``.
        - :class:`str` - updates :attr:`adata` with the following fields:

          - :attr:`obsm['{key_added}'] <anndata.AnnData.obsm>` - the aligned spatial coordinates.
          - :attr:`uns['{key_added}']['alignment_metadata'] <anndata.AnnData.uns>` - the metadata.
        """
        if isinstance(self._policy, StarPolicy):
            reference = self._policy.reference
            logger.debug(f"Setting `reference={reference}`.")
        if reference is None:
            raise ValueError("Please specify `reference=...`.")
        if reference not in self._policy._cat:
            raise ValueError(f"Reference `{reference}` is not in policy's categories `{self._policy._cat}`.")

        if spatial_key is None:
            spatial_key = self.spatial_key

        aligned_maps, aligned_metadata = self._interpolate_scheme(
            reference=reference, mode=mode, spatial_key=spatial_key  # type: ignore[arg-type]
        )

        batch_categories = self.adata.obs[self._policy.key].cat.categories
        batch_indices = {}
        result = np.zeros((self.adata.n_obs, aligned_maps[list(aligned_maps.keys())[0]].shape[1]))

        # Create batch to index mapping
        for cat in batch_categories:
            batch_indices[cat] = np.where(self.adata.obs[self._policy.key] == cat)[0]

        # Assign aligned coordinates to the correct positions in result array
        for cat, indices in batch_indices.items():
            if cat in aligned_maps:
                result[indices] = aligned_maps[cat]

        if key_added is None:
            return result, aligned_metadata

        self.adata.obsm[key_added] = result
        if mode == "affine":  # noqa: RET503
            self.adata.uns.setdefault(key_added, {})
            self.adata.uns[key_added]["alignment_metadata"] = aligned_metadata

        return None

    def cell_transition(
        self,
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
            - See :doc:`../../notebooks/examples/plotting/200_cell_transitions` on how to
              compute and plot the cell transitions.

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

        return self._cell_transition(
            key=self.batch_key,
            source=source,
            target=target,
            source_groups=source_groups,
            target_groups=target_groups,
            forward=forward,
            aggregation_mode=aggregation_mode,
            other_key=None,
            other_adata=None,
            batch_size=batch_size,
            normalize=normalize,
            key_added=key_added,
        )

    def annotation_mapping(
        self,
        mapping_mode: Literal["sum", "max"],
        annotation_label: str,
        forward: bool,
        source: K = "src",
        target: K = "tgt",
        batch_size: Optional[int] = None,
        cell_transition_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        other_adata: Optional[AnnData] = None,
        scale_by_marginals: bool = True,
    ) -> pd.DataFrame:
        """Transfer annotations between distributions.

        This function transfers annotations (e.g. cell type labels) between distributions of cells.

        Parameters
        ----------
        mapping_mode
            How to decide which label to transfer. Valid options are:

            - ``'max'`` - pick the label of the annotated cell with the highest matching probability.
            - ``'sum'`` - aggregate the annotated cells by label then
              pick the label with the highest total matching probability.
        annotation_label
            Key in :attr:`~anndata.AnnData.obs` where the annotation is stored.
        forward
            If :obj:`True`, transfer the annotations from ``source`` to ``target``.
        source
            Key identifying the source distribution.
        target
            Key identifying the target distribution.
        batch_size
            Number of rows/columns of the cost matrix to materialize during :meth:`push` or :meth:`pull`.
            Larger value will require more memory.
            If :obj:`None`, the entire cost matrix will be materialized.
        cell_transition_kwargs
            Keyword arguments for :meth:`cell_transition`, used only if ``mapping_mode = 'sum'``.
        other_adata
            Other adata object to use for the cell transitions.
        scale_by_marginals
            Whether to scale by the source/target :term:`marginals`.

        Returns
        -------
        :class:`~pandas.DataFrame` - Returns the DataFrame of transferred annotations.
        """
        return self._annotation_mapping(
            mapping_mode=mapping_mode,
            annotation_label=annotation_label,
            source=source,
            target=target,
            key=self.batch_key,
            forward=forward,
            batch_size=batch_size,
            cell_transition_kwargs=cell_transition_kwargs,
            other_adata=other_adata,
            scale_by_marginals=scale_by_marginals,
        )

    @property
    def spatial_key(self) -> Optional[str]:
        """Spatial key in :attr:`~anndata.AnnData.obsm`."""
        return self._spatial_key

    @spatial_key.setter
    def spatial_key(self, key: Optional[str]) -> None:
        if key is not None and key not in self.adata.obsm:
            raise KeyError(f"Unable to find spatial data in `adata.obsm[{key!r}]`.")
        self._spatial_key = key

    @property
    def batch_key(self) -> Optional[str]:
        """Batch key in :attr:`~anndata.AnnData.obs`."""
        return self._batch_key

    @batch_key.setter
    def batch_key(self, key: Optional[str]) -> None:
        if key is not None and key not in self.adata.obs:
            raise KeyError(f"Unable to find batch data in `adata.obs[{key!r}]`.")
        self._batch_key = key

    def _subset_spatial(
        self,
        k: K,
        spatial_key: str,
    ) -> ArrayLike:
        mask = self.adata.obs[self._policy.key] == k
        return self.adata[mask].obsm[spatial_key].astype(float, copy=True)


class SpatialMappingMixin(AnalysisMixin[K, B], AbstractSpSc):
    """Spatial mapping analysis mixin class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._batch_key: Optional[str] = None
        self._spatial_key: Optional[str] = None

    def _filter_vars(
        self,
        var_names: Optional[Sequence[str]] = None,
    ) -> Optional[List[str]]:
        """Filter variables for the linear term."""
        vars_sc = set(self.adata_sc.var_names)  # TODO: allow alternative gene symbol by passing var_key
        vars_sp = set(self.adata_sp.var_names)
        if var_names is None:
            var_names = vars_sc & vars_sp
            if var_names:
                return list(var_names)
            raise ValueError("Single-cell and spatial `AnnData` do not share any variables.")

        var_names = set(var_names)
        if not var_names:
            return None

        if var_names.issubset(vars_sc) and var_names.issubset(vars_sp):  # type: ignore[attr-defined]
            return list(var_names)

        raise ValueError("Some variable are missing in the single-cell or the spatial `AnnData`.")

    def correlate(
        self,
        var_names: Optional[Sequence[str]] = None,
        corr_method: Literal["pearson", "spearman"] = "pearson",
        device: Optional[Device_t] = None,
        groupby: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> Union[Mapping[Tuple[K, K], Mapping[Any, pd.Series]], Mapping[Tuple[K, K], pd.Series]]:
        """Correlate true and predicted gene expression.

        .. warning::
            Sparse matrices stored in :attr:`~anndata.AnnData.X` will be densified.

        Parameters
        ----------
        var_names
            Keys in :attr:`~anndata.AnnData.var_names`. If :obj:`None`, use all shared genes.
        corr_method
            Correlation method. Valid options are:

                - ``'pearson'`` - `Pearson correlation <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_.
                - ``'spearman'`` - `Spearman rank correlation <https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient>`_.

        device
            Device where to transfer the solutions, see :meth:`~moscot.base.output.BaseDiscreteSolverOutput.to`.
        groupby
            Optional key in :attr:`~anndata.AnnData.obs`, containing categorical annotations for grouping.
        batch_size
            Number of features to process at once. If :obj:`None`, process all features at once.
            Larger values will require more memory.

        Returns
        -------
        Correlation for each solution in `solutions`.
        """
        var_sc = self._filter_vars(var_names)
        if var_sc is None or not len(var_sc):
            raise ValueError("No overlapping `var_names` between spatial and gene expression data.")

        if corr_method == "pearson":
            corr = st.pearsonr
        elif corr_method == "spearman":
            corr = st.spearmanr
        else:
            raise NotImplementedError(f"Correlation method `{corr_method!r}` is not yet implemented.")

        gexp_sc = self.adata_sc[:, var_sc].X
        if sp.issparse(gexp_sc):
            gexp_sc = gexp_sc.toarray()

        corrs: Union[Dict[Tuple[K, K], Dict[Any, pd.Series]], Dict[Tuple[K, K], pd.Series]] = {}
        for key, val in self.solutions.items():
            # create mask corresponding to the current batch of spatial data
            index_obs = (
                (self.adata_sp.obs[self._policy.key] == key[0])
                if self._policy.key is not None
                else np.arange(self.adata_sp.shape[0])
            )
            adata_sp = self.adata_sp[index_obs, var_sc]

            # initialize a dict of group masks
            if groupby:
                groups = adata_sp.obs[groupby].cat.categories
                group_masks = {group: (adata_sp.obs[groupby]).values == group for group in groups}
                corrs[key] = {}
            else:
                group_masks = {"all": np.ones(adata_sp.shape[0], dtype=bool)}

            gexp_sp = adata_sp.X
            if sp.issparse(gexp_sp):
                gexp_sp = gexp_sp.toarray()

            # predict spatial feature expression
            n_splits = np.max([np.floor(gexp_sc.shape[1] / batch_size), 1]) if batch_size else 1
            logger.debug(f"Processing {gexp_sc.shape[1]} features in {n_splits} batches.")
            gexp_pred_sp = np.hstack(
                [
                    val.to(device=device).pull(x, scale_by_marginals=True)
                    for x in np.array_split(gexp_sc, n_splits, axis=1)
                ],
            )

            # loop over groups and compute correlations
            for group, group_mask in group_masks.items():
                if np.sum(group_mask) < 2:
                    logger.debug(f"Skipping `group={group}` as it contains less then 2 samples.")
                    continue

                corr_val = [
                    corr(gexp_pred_sp[group_mask, gi], gexp_sp[group_mask, gi])[0] for gi, _ in enumerate(var_sc)
                ]
                if groupby:
                    corrs[key][group] = pd.Series(corr_val, index=var_sc)
                else:
                    corrs[key] = pd.Series(corr_val, index=var_sc)

        return corrs

    def impute(
        self,
        var_names: Optional[Sequence[str]] = None,
        device: Optional[Device_t] = None,
        batch_size: Optional[int] = None,
    ) -> AnnData:
        """Impute the expression of specific genes.

        Parameters
        ----------
        var_names
            Genes in :attr:`~anndata.AnnData.var_names` to impute. If :obj:`None`, use all genes in :attr:`adata_sc`.
        device
            Device where to transfer the solutions, see :meth:`~moscot.base.output.BaseDiscreteSolverOutput.to`.
        batch_size:
            Number of features to process at once. If :obj:`None`, process all features at once.
            Larger values will require more memory.

        Returns
        -------
        Annotated data object with the imputed gene expression.
        """
        if var_names is None:
            var_names = self.adata_sc.var_names

        gexp_sc = self.adata_sc[:, var_names].X
        if sp.issparse(gexp_sc):
            gexp_sc = gexp_sc.toarray()

        # predict spatial feature expression
        n_splits = np.max([np.floor(gexp_sc.shape[1] / batch_size), 1]) if batch_size else 1
        logger.debug(f"Processing {gexp_sc.shape[1]} features in {n_splits} batches.")

        predictions = np.nan_to_num(
            np.vstack(
                [
                    np.hstack(
                        [
                            val.to(device=device).pull(x, scale_by_marginals=True)
                            for x in np.array_split(gexp_sc, n_splits, axis=1)
                        ]
                    )
                    for val in self.solutions.values()
                ]
            ),
            nan=0.0,
            copy=False,
        )

        adata_pred = AnnData(X=predictions, obsm=self.adata_sp.obsm.copy())
        adata_pred.obs_names = self.adata_sp.obs_names
        adata_pred.var_names = var_names

        return adata_pred

    def spatial_correspondence(
        self,
        interval: Union[int, ArrayLike] = 10,
        max_dist: Optional[int] = None,
        attr: Optional[Dict[str, Optional[str]]] = None,
    ) -> pd.DataFrame:
        """Compute structural correspondence between spatial and molecular distances.

        Parameters
        ----------
        interval
            Interval for the spatial distance. If :class:`int`, it will be set from the data.
        max_dist
            Maximum distance for the interval. If :obj:`None`, it will set from the data.
        attr
            How to extract the data for correspondence. Valid options are:

            - :obj:`None` - use :attr:`~anndata.AnnData.X`.
            - :class:`dict` - key corresponds to an attribute of :class:`~anndata.AnnData` and
              value to a key in that attribute. If the value is :obj:`None`, only the attribute will be used.

        Returns
        -------
        A dataframe with the following columns:

        - ``'features_distance'`` - average spatial distance.
        - ``'index_interval'`` - index of the interval.
        - ``'value_interval'`` - average expression distance.
        - ``'{batch_key}'`` key of the batch (slide).
        """

        def _get_features(
            adata: AnnData,
            attr: Optional[Dict[str, Any]] = None,
        ) -> ArrayLike:
            attr = {"attr": "X"} if attr is None else attr
            att = attr.get("attr", None)
            key = attr.get("key", None)

            if key is not None:
                return getattr(adata, att)[key]
            return getattr(adata, att)

        if self.batch_key is None:
            spatial = self.adata.obsm[self.spatial_key]
            features = _get_features(self.adata, attr)
            return _compute_correspondence(spatial, features, interval, max_dist)

        res = []
        for c in self.adata.obs[self.batch_key].cat.categories:
            adata_subset = self.adata[self.adata.obs[self.batch_key] == c]
            spatial = adata_subset.obsm[self.spatial_key]
            features = _get_features(adata_subset, attr)
            out = _compute_correspondence(spatial, features, interval, max_dist)
            out[self.batch_key] = c
            res.append(out)

        res = pd.concat(res, axis=0)
        res[self.batch_key] = res[self.batch_key].astype("category")  # type: ignore[call-overload]
        return res

    def cell_transition(
        self,
        source: K,
        target: Optional[K] = None,
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
            - See :doc:`../../notebooks/examples/plotting/200_cell_transitions`
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
            assert self.batch_key is not None
        return self._cell_transition(
            key=self.batch_key,
            source=source,
            target=target or None,
            source_groups=source_groups,
            target_groups=target_groups,
            forward=forward,
            aggregation_mode=aggregation_mode,
            other_key=None,
            other_adata=self.adata_sc,
            batch_size=batch_size,
            normalize=normalize,
            key_added=key_added,
        )

    def annotation_mapping(
        self,
        mapping_mode: Literal["sum", "max"],
        annotation_label: str,
        source: K,
        target: K = "tgt",
        forward: bool = False,
        batch_size: Optional[int] = None,
        cell_transition_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
        scale_by_marginals: bool = True,
    ) -> pd.DataFrame:
        """Transfer annotations between distributions.

        This function transfers annotations (e.g. cell type labels) between distributions of cells.

        Parameters
        ----------
        mapping_mode
            How to decide which label to transfer. Valid options are:

            - ``'max'`` - pick the label of the annotated cell with the highest matching probability.
            - ``'sum'`` - aggregate the annotated cells by label then
              pick the label with the highest total matching probability.
        annotation_label
            Key in :attr:`~anndata.AnnData.obs` where the annotation is stored.
        forward
            If :obj:`True`, transfer the annotations from ``source`` to ``target``.
        source
            Key identifying the source distribution.
        target
            Key identifying the target distribution.
        batch_size
            Number of rows/columns of the cost matrix to materialize during :meth:`push` or :meth:`pull`.
            Larger value will require more memory.
            If :obj:`None`, the entire cost matrix will be materialized.
        cell_transition_kwargs
            Keyword arguments for :meth:`cell_transition`, used only if ``mapping_mode = 'sum'``.
        scale_by_marginals
            Whether to scale by the source/target :term:`marginals`.

        Returns
        -------
        :class:`~pandas.DataFrame` - Returns the DataFrame of transferred annotations.
        """
        return self._annotation_mapping(
            mapping_mode=mapping_mode,
            annotation_label=annotation_label,
            source=source,
            target=target,
            forward=forward,
            key=self.batch_key,
            other_adata=self.adata_sc,
            batch_size=batch_size,
            cell_transition_kwargs=cell_transition_kwargs,
            scale_by_marginals=scale_by_marginals,
        )

    @property
    def batch_key(self) -> Optional[str]:
        """Batch key in :attr:`~anndata.AnnData.obs`."""
        return self._batch_key

    @batch_key.setter
    def batch_key(self, key: Optional[str]) -> None:
        if key is not None and key not in self.adata.obs:
            raise KeyError(f"Unable to find batch data in `adata.obs[{key!r}]`.")
        self._batch_key = key

    @property
    def spatial_key(self) -> Optional[str]:
        """Spatial key in :attr:`~anndata.AnnData.obsm`."""
        return self._spatial_key

    @spatial_key.setter
    def spatial_key(self, key: Optional[str]) -> None:
        if key is not None and key not in self.adata.obsm:
            raise KeyError(f"Unable to find spatial data in `adata.obsm[{key!r}]`.")
        self._spatial_key = key


def _compute_correspondence(
    spatial: ArrayLike,
    features: Union[ArrayLike, sp.spmatrix, sp.sparray],
    interval: Union[int, ArrayLike] = 10,
    max_dist: Optional[int] = None,
) -> pd.DataFrame:
    if isinstance(interval, int):
        # prepare support
        hull = ConvexHull(spatial)
        area = hull.volume
        if max_dist is None:
            max_dist = round(((area / 2) ** 0.5) / 2)
        support = np.linspace(max_dist / interval, max_dist, interval)
    else:
        support = np.asarray(np.sort(interval), dtype=float)

    def pdist(row_idx: ArrayLike, col_idx: float, feat: ArrayLike) -> Any:
        if len(row_idx) > 0:
            return pairwise_distances(feat[row_idx, :], feat[[col_idx], :]).mean()  # type: ignore[index]
        return np.nan

    # TODO(michalk8): vectorize using jax, this is just a for loop
    vpdist = np.vectorize(pdist, excluded=["feat"])
    if sp.issparse(features):
        features_sp: Union[sp.spmatrix, sp.sparray] = features
        features = features_sp.toarray()

    feat_arr, index_arr, support_arr = [], [], []
    for ind, i in enumerate(support):
        tree = NearestNeighbors(radius=i).fit(spatial)
        _, idx = tree.radius_neighbors()

        feat_dist = vpdist(row_idx=idx, col_idx=np.arange(len(idx)), feat=features)
        feat_dist = feat_dist[~np.isnan(feat_dist)]

        feat_arr.append(feat_dist)
        index_arr.append(np.repeat(ind, feat_dist.shape[0]))
        support_arr.append(np.repeat(i, feat_dist.shape[0]))

    feat_arr = np.concatenate(feat_arr)
    index_arr = np.concatenate(index_arr)
    support_arr = np.concatenate(support_arr)

    df = pd.DataFrame(
        np.vstack([feat_arr, index_arr, support_arr]).T,
        columns=["features_distance", "index_interval", "value_interval"],
    )

    df["index_interval"] = df["index_interval"].astype(int).astype("category")
    return df


def _affine(
    tmap: sp.linalg.LinearOperator,
    src: ArrayLike,
    tgt: ArrayLike,
) -> Tuple[ArrayLike, ArrayLike]:
    tgt -= tgt.mean(0)
    out = tmap @ src
    H = tgt.T.dot(out)
    U, _, Vt = svd(H)
    R = Vt.T.dot(U.T)
    tgt = R.dot(tgt.T).T
    return tgt, R


def _warp(tmap: sp.linalg.LinearOperator, src: ArrayLike, tgt: ArrayLike) -> Tuple[ArrayLike, None]:
    del tgt
    return tmap @ src, None
