import itertools
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

import numpy as np
import pandas as pd
import scipy.sparse as sp
from networkx import NetworkXNoPath
from pandas.api.types import is_categorical_dtype
from scipy.linalg import svd
from scipy.sparse.linalg import LinearOperator
from scipy.spatial import ConvexHull
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from anndata import AnnData

from moscot import _constants
from moscot._docs._docs import d
from moscot._docs._docs_mixins import d_mixins
from moscot._logging import logger
from moscot._types import ArrayLike, Device_t, Str_Dict_t
from moscot.base.problems._mixins import AnalysisMixin, AnalysisMixinProtocol
from moscot.base.problems.compound_problem import B, K
from moscot.utils.subset_policy import StarPolicy

__all__ = ["SpatialAlignmentMixin", "SpatialMappingMixin"]


class SpatialAlignmentMixinProtocol(AnalysisMixinProtocol[K, B]):
    """Protocol class."""

    spatial_key: Optional[str]
    _spatial_key: Optional[str]
    batch_key: Optional[str]

    def _subset_spatial(  # type:ignore[empty-body]
        self: "SpatialAlignmentMixinProtocol[K, B]",
        k: K,
        spatial_key: Optional[str] = None,
    ) -> ArrayLike:
        ...

    def _interpolate_scheme(  # type:ignore[empty-body]
        self: "SpatialAlignmentMixinProtocol[K, B]",
        reference: K,
        mode: Literal["warp", "affine"],
        spatial_key: Optional[str] = None,
    ) -> Tuple[Dict[K, ArrayLike], Optional[Dict[K, Optional[ArrayLike]]]]:
        ...

    @staticmethod
    def _affine(  # type:ignore[empty-body]
        tmap: LinearOperator, src: ArrayLike, tgt: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        ...

    @staticmethod
    def _warp(  # type: ignore[empty-body]
        tmap: LinearOperator, src: ArrayLike, _: ArrayLike
    ) -> Tuple[ArrayLike, Optional[ArrayLike]]:
        ...

    def _cell_transition(
        self: AnalysisMixinProtocol[K, B],
        *args: Any,
        **kwargs: Any,
    ) -> pd.DataFrame:
        ...


class SpatialMappingMixinProtocol(AnalysisMixinProtocol[K, B]):
    """Protocol class."""

    adata_sc: AnnData
    adata_sp: AnnData
    batch_key: Optional[str]
    spatial_key: Optional[str]
    _spatial_key: Optional[str]

    def _filter_vars(
        self: "SpatialMappingMixinProtocol[K, B]",
        var_names: Optional[Sequence[str]] = None,
    ) -> Optional[List[str]]:
        ...

    def _cell_transition(self: AnalysisMixinProtocol[K, B], *args: Any, **kwargs: Any) -> pd.DataFrame:
        ...


class SpatialAlignmentMixin(AnalysisMixin[K, B]):
    """Spatial alignment mixin class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._spatial_key: Optional[str] = None
        self._batch_key: Optional[str] = None

    def _interpolate_scheme(  # type: ignore[misc]
        self: SpatialAlignmentMixinProtocol[K, B],
        reference: K,
        mode: Literal["warp", "affine"],
        spatial_key: Optional[str] = None,
    ) -> Tuple[Dict[K, ArrayLike], Optional[Dict[K, Optional[ArrayLike]]]]:
        """Scheme for interpolation."""
        # get reference
        src = self._subset_spatial(reference, spatial_key=spatial_key)
        transport_maps: Dict[K, ArrayLike] = {reference: src}
        transport_metadata: Dict[K, Optional[ArrayLike]] = {}
        if mode == "affine":
            src -= src.mean(0)
            transport_metadata = {reference: np.diag((1, 1))}  # 2d data

        # get policy
        reference_ = [reference] if isinstance(reference, str) else reference
        full_steps = self._policy._graph
        starts = set(itertools.chain.from_iterable(full_steps)) - set(reference_)  # type: ignore[call-overload]

        if mode == "affine":
            _transport = self._affine
        elif mode == "warp":
            _transport = self._warp
        else:
            raise NotImplementedError(f"Alignment mode `{mode!r}` is not yet implemented.")

        steps = {}
        for start in starts:
            try:
                steps[start, reference, True] = self._policy.plan(start=start, end=reference)
            except NetworkXNoPath:
                steps[reference, start, False] = self._policy.plan(start=reference, end=start)

        for (start, end, forward), path in steps.items():
            tmap = self._interpolate_transport(path=path, scale_by_marginals=True)
            # make `tmap` to have shape `(m, n_ref)` and apply it to `src` of shape `(n_ref, d)`
            key, tmap = (start, tmap) if forward else (end, tmap.T)
            spatial_data = self._subset_spatial(key, spatial_key=spatial_key)
            transport_maps[key], transport_metadata[key] = _transport(tmap, src=src, tgt=spatial_data)

        return transport_maps, (transport_metadata if mode == "affine" else None)

    @d.dedent
    def align(  # type: ignore[misc]
        self: SpatialAlignmentMixinProtocol[K, B],
        reference: K,
        mode: Literal["warp", "affine"] = "warp",
        spatial_key: Optional[str] = None,
        inplace: bool = True,
    ) -> Optional[Union[ArrayLike, Tuple[ArrayLike, Optional[Dict[K, Optional[ArrayLike]]]]]]:
        """
        Align spatial data.

        Parameters
        ----------
        reference
            Reference key.
        mode
            Alignment mode:

                - "warp": warp the data to the reference.
                - "affine": align the data to the reference using affine transformation.

        %(inplace)s

        Returns
        -------
        %(alignment_mixin_returns)s
        """
        if reference not in self._policy._cat:
            raise ValueError(f"Reference `{reference}` is not in policy's categories: `{self._policy._cat}`.")
        if isinstance(self._policy, StarPolicy) and reference != self._policy.reference:
            # TODO(michalk8): just warn + optional reference?
            raise ValueError(f"Expected reference to be `{self._policy.reference}`, found `{reference}`.")
        aligned_maps, aligned_metadata = self._interpolate_scheme(
            reference=reference, mode=mode, spatial_key=spatial_key
        )
        aligned_basis = np.vstack([aligned_maps[k] for k in self._policy._cat])
        if mode == "affine":
            if not inplace:
                return aligned_basis, aligned_metadata
            if self.spatial_key not in self.adata.uns:
                self.adata.uns[self.spatial_key] = {}
            self.adata.uns[self.spatial_key]["alignment_metadata"] = aligned_metadata
        if not inplace:
            return aligned_basis
        self.adata.obsm[f"{self.spatial_key}_{mode}"] = aligned_basis  # noqa: RET503

    @d_mixins.dedent
    def cell_transition(  # type: ignore[misc]
        self: SpatialAlignmentMixinProtocol[K, B],
        source: K,
        target: K,
        source_groups: Optional[Str_Dict_t] = None,
        target_groups: Optional[Str_Dict_t] = None,
        forward: bool = False,  # return value will be row-stochastic if forward=True, else column-stochastic
        aggregation_mode: Literal["annotation", "cell"] = "annotation",
        batch_size: Optional[int] = None,
        normalize: bool = True,
        key_added: Optional[str] = _constants.CELL_TRANSITION,
    ) -> pd.DataFrame:
        """
        Compute a grouped cell transition matrix.

        This function computes a transition matrix with entries corresponding to categories, e.g. cell types.
        The transition matrix will be row-stochastic if `forward` is `True`, otherwise column-stochastic.

        Parameters
        ----------
        %(cell_trans_params)s
        %(forward_cell_transition)s
        %(aggregation_mode)s
        %(ott_jax_batch_size)s
        %(normalize)s
        %(key_added_plotting)s

        Returns
        -------
        %(return_cell_transition)s

        Notes
        -----
        %(notes_cell_transition)s
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

    @property
    def spatial_key(self) -> Optional[str]:
        """Spatial key in :attr:`anndata.AnnData.obsm`."""
        return self._spatial_key

    @spatial_key.setter
    def spatial_key(self: SpatialAlignmentMixinProtocol[K, B], key: Optional[str]) -> None:  # type: ignore[misc]
        if key is not None and key not in self.adata.obsm:
            raise KeyError(f"Unable to find spatial data in `adata.obsm[{key!r}]`.")
        self._spatial_key = key

    @property
    def batch_key(self) -> Optional[str]:
        """Batch key in :attr:`anndata.AnnData.obs`."""
        return self._batch_key

    @batch_key.setter
    def batch_key(self, key: Optional[str]) -> None:
        if key is not None and key not in self.adata.obs:  # type: ignore[attr-defined]
            raise KeyError(f"Unable to find batch data in `adata.obs[{key!r}]`.")
        self._batch_key = key

    def _subset_spatial(  # type: ignore[misc]
        self: SpatialAlignmentMixinProtocol[K, B], k: K, spatial_key: Optional[str] = None
    ) -> ArrayLike:
        if spatial_key is None:
            spatial_key = self.spatial_key
        return self.adata[self.adata.obs[self._policy._subset_key] == k].obsm[spatial_key].astype(float, copy=True)

    @staticmethod
    def _affine(
        tmap: LinearOperator,
        src: ArrayLike,
        tgt: ArrayLike,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Affine transformation."""
        tgt -= tgt.mean(0)
        out = tmap @ src
        H = tgt.T.dot(out)
        U, _, Vt = svd(H)
        R = Vt.T.dot(U.T)
        tgt = R.dot(tgt.T).T
        return tgt, R

    @staticmethod
    def _warp(tmap: LinearOperator, src: ArrayLike, tgt: ArrayLike) -> Tuple[ArrayLike, None]:
        """Warp transformation."""
        del tgt
        return tmap @ src, None


class SpatialMappingMixin(AnalysisMixin[K, B]):
    """Spatial mapping analysis mixin class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._batch_key: Optional[str] = None
        self._spatial_key: Optional[str] = None

    def _filter_vars(  # type: ignore[misc]
        self: SpatialMappingMixinProtocol[K, B],
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

    def correlate(  # type: ignore[misc]
        self: SpatialMappingMixinProtocol[K, B],
        var_names: Optional[List[str]] = None,
        corr_method: Literal["pearson", "spearman"] = "pearson",
    ) -> Mapping[Tuple[K, K], pd.Series]:
        """
        Calculate correlation between true and predicted gene expression.

        Parameters
        ----------
        var_names
            List of variable names.
        corr_method
            Correlation method:

                - 'pearson': Pearson correlation.
                - 'spearman': Spearman correlation

        Returns
        -------
        TODO
        """
        var_sc = self._filter_vars(var_names)
        if var_sc is None or not len(var_sc):
            raise ValueError("No overlapping `var_names` between ` adata_sc` and `adata_sp`.")

        if corr_method == "pearson":
            cor = pearsonr
        elif corr_method == "spearman":
            cor = spearmanr
        else:
            raise NotImplementedError(f"Correlation method `{corr_method!r}` is not yet implemented.")

        corrs = {}
        gexp_sc = self.adata_sc[:, var_sc].X if not sp.issparse(self.adata_sc.X) else self.adata_sc[:, var_sc].X.A
        for key, val in self.solutions.items():
            index_obs: List[Union[bool, int]] = (
                self.adata_sp.obs[self._policy._subset_key] == key[0]
                if self._policy._subset_key is not None
                else np.arange(self.adata_sp.shape[0])
            )
            gexp_sp = self.adata_sp[index_obs, var_sc].X
            if sp.issparse(gexp_sp):
                # TODO(giovp): in the future, logg if too large
                gexp_sp = gexp_sp.A
            gexp_pred_sp = val.pull(gexp_sc, scale_by_marginals=True)
            corr_val = [cor(gexp_pred_sp[:, gi], gexp_sp[:, gi])[0] for gi, _ in enumerate(var_sc)]
            corrs[key] = pd.Series(corr_val, index=var_sc)

        return corrs

    def impute(  # type: ignore[misc]
        self: SpatialMappingMixinProtocol[K, B],
        var_names: Optional[Sequence[Any]] = None,
        device: Optional[Device_t] = None,
    ) -> AnnData:
        """Impute expression of specific genes.

        Parameters
        ----------
        var_names:
            TODO: don't use device from docstrings here, as different use

        Returns
        -------
        Annotated data object with imputed gene expression values.
        """
        if var_names is None:
            var_names = self.adata_sc.var_names
        gexp_sc = self.adata_sc[:, var_names].X if not sp.issparse(self.adata_sc.X) else self.adata_sc[:, var_names].X.A
        pred_list = [
            val.to(device=device).pull(gexp_sc, scale_by_marginals=True)
            if device is not None
            else val.pull(gexp_sc, scale_by_marginals=True)
            for val in self.solutions.values()
        ]
        gexp_pred = np.nan_to_num(np.vstack(pred_list), nan=0.0, copy=False)
        adata_pred = AnnData(gexp_pred, dtype=np.float_)
        adata_pred.obs_names = self.adata_sp.obs_names
        adata_pred.var_names = var_names
        adata_pred.obsm = self.adata_sp.obsm.copy()
        return adata_pred

    @d.dedent
    def spatial_correspondence(  # type: ignore[misc]
        self: SpatialMappingMixinProtocol[K, B],
        interval: Union[ArrayLike, int] = 10,
        max_dist: Optional[int] = None,
        attr: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Compute structural correspondence between spatial and molecular distances.

        Parameters
        ----------
        interval
            Interval for the spatial distance.
        max_dist
            Maximum distance for the interval, if `None` it is set from data.
        attr
            Specify the attributes from which to compute the correspondence.

        Returns
        -------
        :class:`pandas.DataFrame` with columns:

            - `spatial`: average spatial distance.
            - `expression`: average expression distance.
            - `index`: index of the interval.
            - `batch_key`: key of the batch (slide).
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

        if self.batch_key is not None:
            out_list = []
            if is_categorical_dtype(self.adata.obs[self.batch_key]):
                categ = self.adata.obs[self.batch_key].cat.categories
            else:
                logger.info(f"adata_sp.obs[`{self.batch_key}`] is not `categorical`, using `unique()` method.")
                categ = self.adata.obs[self.batch_key].unique()
            if len(categ) > 1:
                for c in categ:
                    adata_subset = self.adata[self.adata.obs[self.batch_key] == c]
                    spatial = adata_subset.obsm[self.spatial_key]
                    features = _get_features(adata_subset, attr)
                    out = _compute_correspondence(spatial, features, interval, max_dist)
                    out[self.batch_key] = c
                    out_list.append(out)
            else:
                spatial = self.adata.obsm[self.spatial_key]
                features = _get_features(self.adata, attr)
                out = _compute_correspondence(spatial, features, interval, max_dist)
                out[self.batch_key] = categ[0]
                out_list.append(out)
            out = pd.concat(out_list, axis=0)
            out[self.batch_key] = pd.Categorical(out[self.batch_key])
            return out

        spatial = self.adata.obsm[self.spatial_key]
        features = _get_features(self.adata, attr)
        return _compute_correspondence(spatial, features, interval, max_dist)

    @d_mixins.dedent
    def cell_transition(  # type: ignore[misc]
        self: SpatialMappingMixinProtocol[K, B],
        source: K,
        target: Optional[K] = None,
        source_groups: Optional[Str_Dict_t] = None,
        target_groups: Optional[Str_Dict_t] = None,
        forward: bool = False,  # return value will be row-stochastic if forward=True, else column-stochastic
        aggregation_mode: Literal["annotation", "cell"] = "annotation",
        batch_size: Optional[int] = None,
        normalize: bool = True,
        key_added: Optional[str] = _constants.CELL_TRANSITION,
    ) -> pd.DataFrame:
        """
        Compute a grouped cell transition matrix.

        This function computes a transition matrix with entries corresponding to categories, e.g. cell types.
        The transition matrix will be row-stochastic if `forward` is `True`, otherwise column-stochastic.

        Parameters
        ----------
        %(cell_trans_params)s
        %(forward_cell_transition)s
        %(aggregation_mode)s
        %(ott_jax_batch_size)s
        %(normalize)s
        %(key_added_plotting)s

        Returns
        -------
        %(return_cell_transition)s

        Notes
        -----
        %(notes_cell_transition)s
        """
        if TYPE_CHECKING:
            assert self.batch_key is not None
        return self._cell_transition(
            key=self.batch_key,
            source=source,
            target=target,
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

    @property
    def batch_key(self) -> Optional[str]:
        """Batch key in :attr:`anndata.AnnData.obs`."""
        return self._batch_key

    @batch_key.setter
    def batch_key(self, key: Optional[str]) -> None:
        if key is not None and key not in self.adata.obs:  # type: ignore[attr-defined]
            raise KeyError(f"Unable to find batch data in `adata.obs[{key!r}]`.")
        self._batch_key = key

    @property
    def spatial_key(self) -> Optional[str]:
        """Spatial key in :attr:`anndata.AnnData.obsm`."""
        return self._spatial_key

    @spatial_key.setter
    def spatial_key(self: SpatialAlignmentMixinProtocol[K, B], key: Optional[str]) -> None:  # type: ignore[misc]
        if key is not None and key not in self.adata.obsm:
            raise KeyError(f"Unable to find spatial data in `adata.obsm[{key!r}]`.")
        self._spatial_key = key


def _compute_correspondence(
    spatial: ArrayLike,
    features: ArrayLike,
    interval: Union[ArrayLike, int] = 10,
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
        support = np.array(sorted(interval), dtype=np.float_, copy=True)

    def pdist(row_idx: ArrayLike, col_idx: float, feat: ArrayLike) -> Any:
        if len(row_idx) > 0:
            return pairwise_distances(feat[row_idx, :], feat[[col_idx], :]).mean()  # type: ignore[index]
        return np.nan

    vpdist = np.vectorize(pdist, excluded=["feat"])
    features = features.A if sp.issparse(features) else features  # type: ignore[attr-defined]

    feat_arr = []
    index_arr = []
    support_arr = []

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

    df["index_interval"] = pd.Categorical(df["index_interval"].astype(np.int_))
    return df
