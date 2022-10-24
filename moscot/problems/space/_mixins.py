from typing import Any, Dict, List, Tuple, Union, Literal, Mapping, Optional, Sequence, TYPE_CHECKING
from itertools import chain

from networkx import NetworkXNoPath
from scipy.stats import pearsonr, spearmanr
from scipy.linalg import svd
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator
import pandas as pd

import numpy as np

from anndata import AnnData

from moscot._types import Device_t, ArrayLike, Str_Dict_t
from moscot._docs._docs import d
from moscot.problems.base import AnalysisMixin  # type: ignore[attr-defined]
from moscot._docs._docs_mixins import d_mixins
from moscot._constants._constants import CorrMethod, AlignmentMode, PlottingDefaults
from moscot.problems.base._mixins import AnalysisMixinProtocol
from moscot.problems._subset_policy import StarPolicy
from moscot.problems.base._compound_problem import B, K


class SpatialAlignmentMixinProtocol(AnalysisMixinProtocol[K, B]):
    """Protocol class."""

    spatial_key: Optional[str]
    _spatial_key: Optional[str]
    batch_key: Optional[str]

    def _subset_spatial(self: "SpatialAlignmentMixinProtocol[K, B]", k: K) -> ArrayLike:
        ...

    def _interpolate_scheme(
        self: "SpatialAlignmentMixinProtocol[K, B]",
        reference: K,
        mode: Literal["warp", "affine"],
    ) -> Tuple[Dict[K, ArrayLike], Optional[Dict[K, Optional[ArrayLike]]]]:
        ...

    @staticmethod
    def _affine(tmap: LinearOperator, tgt: ArrayLike, src: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        ...

    @staticmethod
    def _warp(tmap: LinearOperator, _: ArrayLike, src: ArrayLike) -> Tuple[ArrayLike, None]:
        ...

    def _cell_transition(  # TODO(@MUCDK) think about removing _cell_transition_non_online
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

    def _filter_vars(
        self: "SpatialMappingMixinProtocol[K, B]",
        var_names: Optional[Sequence[str]] = None,
    ) -> Optional[List[str]]:
        ...

    def _cell_transition(  # TODO(@MUCDK) think about removing _cell_transition_non_online
        self: AnalysisMixinProtocol[K, B],
        *args: Any,
        **kwargs: Any,
    ) -> pd.DataFrame:
        ...


class SpatialAlignmentMixin(AnalysisMixin[K, B]):
    """Spatial alignment mixin class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._spatial_key: Optional[str] = None
        self._batch_key: Optional[str] = None

    def _interpolate_scheme(
        self: SpatialAlignmentMixinProtocol[K, B],
        reference: K,
        mode: Literal["warp", "affine"],
    ) -> Tuple[Dict[K, ArrayLike], Optional[Dict[K, Optional[ArrayLike]]]]:
        """Scheme for interpolation."""

        # get reference
        src = self._subset_spatial(reference)
        transport_maps: Dict[K, ArrayLike] = {reference: src}
        transport_metadata: Dict[K, Optional[ArrayLike]] = {}
        if mode == "affine":
            src -= src.mean(0)
            transport_metadata = {reference: np.diag((1, 1))}  # 2d data

        # get policy
        full_steps = self._policy._graph
        starts = set(chain.from_iterable(full_steps)) - set(reference)  # type: ignore[call-overload]
        fwd_steps, bwd_steps = {}, {}
        for start in starts:
            try:
                fwd_steps[(start, reference)] = self._policy.plan(start=start, end=reference)
            except NetworkXNoPath:
                bwd_steps[(reference, start)] = self._policy.plan(start=reference, end=start)

        if mode == AlignmentMode.AFFINE:
            transport = self._affine
        elif mode == AlignmentMode.WARP:
            transport = self._warp
        else:
            raise NotImplementedError("TODO")

        if len(fwd_steps):
            for (start, _), path in fwd_steps.items():
                tmap = self._interpolate_transport(path=path, scale_by_marginals=True, forward=True)
                transport_maps[start], transport_metadata[start] = transport(tmap, self._subset_spatial(start), src)

        if len(bwd_steps):
            for (_, end), path in bwd_steps.items():
                tmap = self._interpolate_transport(path=path, scale_by_marginals=True, forward=False)
                transport_maps[end], transport_metadata[end] = transport(tmap.T, self._subset_spatial(end), src)

        if mode == "affine":
            return transport_maps, transport_metadata
        return transport_maps, None

    @d.dedent
    def align(
        self: SpatialAlignmentMixinProtocol[K, B],
        reference: K,
        mode: Literal["warp", "affine"] = "warp",
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
        mode = AlignmentMode(mode)
        if reference not in self._policy._cat:
            raise ValueError(f"`reference: {reference}` not in policy categories: {self._policy._cat}.")
        if isinstance(self._policy, StarPolicy) and reference != self._policy.reference:
            raise ValueError(f"Invalid `reference: {reference}` for `policy='star'`.")

        aligned_maps, aligned_metadata = self._interpolate_scheme(reference=reference, mode=mode)
        aligned_basis = np.vstack([aligned_maps[k] for k in self._policy._cat])
        if mode == "affine":
            if not inplace:
                return aligned_basis, aligned_metadata
            if self.spatial_key not in self.adata.uns:
                self.adata.uns[self.spatial_key] = {}
            self.adata.uns[self.spatial_key]["alignment_metadata"] = aligned_metadata
        if not inplace:
            return aligned_basis
        self.adata.obsm[f"{self.spatial_key}_{mode}"] = aligned_basis

    @d_mixins.dedent
    def cell_transition(
        self: SpatialAlignmentMixinProtocol[K, B],
        source: K,
        target: K,
        source_groups: Optional[Str_Dict_t] = None,
        target_groups: Optional[Str_Dict_t] = None,
        forward: bool = False,  # return value will be row-stochastic if forward=True, else column-stochastic
        aggregation_mode: Literal["annotation", "cell"] = "annotation",
        online: bool = False,
        batch_size: Optional[int] = None,
        normalize: bool = True,
        key_added: Optional[str] = PlottingDefaults.CELL_TRANSITION,
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
        %(online)s
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
            online=online,
            other_key=None,
            other_adata=None,
            batch_size=batch_size,
            normalize=normalize,
            key_added=key_added,
        )

    @property
    def spatial_key(self) -> Optional[str]:
        """Return spatial key."""
        return self._spatial_key

    @spatial_key.setter
    def spatial_key(self: SpatialAlignmentMixinProtocol[K, B], value: Optional[str]) -> None:
        if value is not None and value not in self.adata.obsm:
            raise KeyError(f"TODO: {value} not found in `adata.obsm`.")
        self._spatial_key = value

    @property
    def batch_key(self) -> Optional[str]:
        """Return batch key."""
        return self._batch_key

    @batch_key.setter
    def batch_key(self, value: Optional[str]) -> None:
        if value is not None and value not in self.adata.obs:
            raise KeyError(f"{value} not in `adata.obs`.")
        self._batch_key = value

    def _subset_spatial(self: SpatialAlignmentMixinProtocol[K, B], k: K) -> ArrayLike:
        return (
            self.adata[self.adata.obs[self._policy._subset_key] == k]
            .obsm[self.spatial_key]
            .astype(np.float_, copy=True)
        )

    @staticmethod
    def _affine(tmap: LinearOperator, tgt: ArrayLike, src: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """Affine transformation."""
        tgt -= tgt.mean(0)
        H = tgt.T.dot(tmap.dot(src))
        U, _, Vt = svd(H)
        R = Vt.T.dot(U.T)
        tgt = R.dot(tgt.T).T
        return tgt, R

    @staticmethod
    def _warp(tmap: LinearOperator, _: ArrayLike, src: ArrayLike) -> Tuple[ArrayLike, None]:
        """Warp transformation."""
        return tmap.dot(src), None


class SpatialMappingMixin(AnalysisMixin[K, B]):
    """Spatial mapping analysis mixin class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._batch_key: Optional[str] = None

    def _filter_vars(
        self: SpatialMappingMixinProtocol[K, B],
        var_names: Optional[Sequence[str]] = None,
    ) -> Optional[List[str]]:
        """Filter variables for Sinkhorn term."""
        vars_sc = set(self.adata_sc.var_names)  # TODO: allow alternative gene symbol by passing var_key
        vars_sp = set(self.adata_sp.var_names)
        _var_names = set(var_names) if var_names is not None else None
        if _var_names is None:
            _var_names = vars_sp.intersection(vars_sc)
            if len(_var_names):
                return list(_var_names)
            raise ValueError("`adata_sc` and `adata_sp` do not share `var_names`. Input valid `var_names`.")
        if not len(_var_names):
            return None
        if _var_names.issubset(vars_sc) and _var_names.issubset(vars_sp):
            return list(_var_names)
        raise ValueError("Some `var_names` ares missing in either `adata_sc` or `adata_sp`.")

    def correlate(
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

        corr_method = CorrMethod(corr_method)
        if corr_method == CorrMethod.PEARSON:
            cor = pearsonr
        elif corr_method == CorrMethod.SPEARMAN:
            cor = spearmanr
        else:
            raise NotImplementedError("TODO: `corr_method` must be `pearson` or `spearman`.")

        corrs = {}
        gexp_sc = self.adata_sc[:, var_sc].X if not issparse(self.adata_sc.X) else self.adata_sc[:, var_sc].X.A
        for key, val in self.solutions.items():
            index_obs: List[Union[bool, int]] = (
                self.adata_sp.obs[self._policy._subset_key] == key[0]
                if self._policy._subset_key is not None
                else np.arange(self.adata_sp.shape[0])
            )
            gexp_sp = self.adata_sp[index_obs, var_sc].X
            if issparse(gexp_sp):
                # TODO(giovp): in future, logg if too large
                gexp_sp = gexp_sp.A
            gexp_pred_sp = val.pull(gexp_sc, scale_by_marginals=True)
            corr_val = [cor(gexp_pred_sp[:, gi], gexp_sp[:, gi])[0] for gi, _ in enumerate(var_sc)]
            corrs[key] = pd.Series(corr_val, index=var_sc)

        return corrs

    def impute(
        self: SpatialMappingMixinProtocol[K, B],
        var_names: Optional[Sequence[Any]] = None,
        device: Optional[Device_t] = None,
    ) -> AnnData:
        """
        Impute expression of specific genes.

        Parameters
        ----------
        TODO: don't use device from docstrings here, as different use

        Returns
        -------
        :class:`anndata.AnnData` with imputed gene expression values.
        """
        if var_names is None:
            var_names = self.adata_sc.var_names
        gexp_sc = self.adata_sc[:, var_names].X if not issparse(self.adata_sc.X) else self.adata_sc[:, var_names].X.A
        pred_list = [
            val.to(device=device).pull(gexp_sc, scale_by_marginals=True)
            if device is not None
            else val.pull(gexp_sc, scale_by_marginals=True)
            for val in self.solutions.values()
        ]
        adata_pred = AnnData(np.nan_to_num(np.vstack(pred_list), nan=0.0, copy=False))
        adata_pred.obs_names = self.adata_sp.obs_names.copy()
        adata_pred.var_names = var_names
        return adata_pred

    @d_mixins.dedent
    def cell_transition(
        self: SpatialMappingMixinProtocol[K, B],
        source: K,
        target: Optional[K] = None,
        source_groups: Optional[Str_Dict_t] = None,
        target_groups: Optional[Str_Dict_t] = None,
        forward: bool = False,  # return value will be row-stochastic if forward=True, else column-stochastic
        aggregation_mode: Literal["annotation", "cell"] = "annotation",
        online: bool = False,
        batch_size: Optional[int] = None,
        normalize: bool = True,
        key_added: Optional[str] = PlottingDefaults.CELL_TRANSITION,
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
        %(online)s
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
            online=online,
            other_key=None,
            other_adata=self.adata_sc,
            batch_size=batch_size,
            normalize=normalize,
            key_added=key_added,
        )

    @property
    def batch_key(self) -> Optional[str]:
        """Return batch key."""
        return self._batch_key

    @batch_key.setter
    def batch_key(self, value: Optional[str]) -> None:
        if value is not None and value not in self.adata.obs:
            raise KeyError(f"{value} not in `adata.obs`.")
        self._batch_key = value
