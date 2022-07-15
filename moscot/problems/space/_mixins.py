from typing import Any, Dict, List, Tuple, Union, Mapping, Callable, Optional, Sequence, TYPE_CHECKING
from itertools import chain

from networkx import NetworkXNoPath
from scipy.stats import pearsonr, spearmanr
from scipy.linalg import svd
from scipy.sparse import issparse
from typing_extensions import Literal
from scipy.sparse.linalg import LinearOperator
import pandas as pd

import numpy as np

from anndata import AnnData

from moscot._docs import d
from moscot._types import Filter_t, ArrayLike
from moscot.problems.base import AnalysisMixin  # type: ignore[attr-defined]
from moscot._constants._constants import CorrMethod, AlignmentMode, AggregationMode
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

    def _affine(tmap: LinearOperator, tgt: ArrayLike, src: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        ...

    def _warp(tmap: LinearOperator, _: ArrayLike, src: ArrayLike) -> Tuple[ArrayLike, None]:
        ...

    def _cell_transition(  # TODO(@MUCDK) think about removing _cell_transition_non_online
        self: AnalysisMixinProtocol[K, B],
        online: bool,
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
        online: bool,
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

        # TODO(giovp): error message for star policy
        # get reference
        src = self._subset_spatial(reference)
        transport_maps: Dict[K, ArrayLike] = {reference: src}
        transport_metadata: Dict[K, Optional[ArrayLike]] = {}
        if mode == "affine":
            # TODO(michalk8): why the diag?
            src -= src.mean(0)
            transport_metadata = {reference: np.diag((1, 1))}  # 2d data

        # get policy
        full_steps = self._policy._graph
        starts = set(chain.from_iterable(full_steps)) - set(reference)  # type: ignore[arg-type]
        fwd_steps, bwd_steps = {}, {}
        for start in starts:
            try:
                fwd_steps[(start, reference)] = self._policy.plan(start=start, end=reference)
            except NetworkXNoPath:
                bwd_steps[(reference, start)] = self._policy.plan(start=reference, end=start)

        if mode == "affine":  # TODO(@MUCDK): infer correct types
            _transport: Callable[
                [LinearOperator, ArrayLike, ArrayLike], Tuple[ArrayLike, Optional[ArrayLike]]
            ] = self._affine  # type: ignore[assignment]
        else:
            _transport = self._warp  # type: ignore[assignment]

        if len(fwd_steps):
            for (start, _), path in fwd_steps.items():
                tmap = self._interpolate_transport(path=path, scale_by_marginals=True, forward=True)
                transport_maps[start], transport_metadata[start] = _transport(tmap, self._subset_spatial(start), src)

        if len(bwd_steps):
            for (_, end), path in bwd_steps.items():
                tmap = self._interpolate_transport(path=path, scale_by_marginals=True, forward=False)
                transport_maps[end], transport_metadata[end] = _transport(tmap.T, self._subset_spatial(end), src)

        if mode == "affine":
            return transport_maps, transport_metadata
        return transport_maps, None

    @d.dedent
    def align(
        self: SpatialAlignmentMixinProtocol[K, B],
        reference: K,
        mode: Literal["warp", "affine"] = AlignmentMode.WARP,  # type: ignore[assignment]
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
            raise ValueError(f"`reference: {reference}` not in policy categories: {self._policy._cat}.")
        if isinstance(self._policy, StarPolicy):
            if reference != self._policy.reference:
                raise ValueError(f"Invalid `reference: {reference}` for `policy='star'`.")
        aligned_maps, aligned_metadata = self._interpolate_scheme(reference=reference, mode=mode)
        aligned_basis = np.vstack([aligned_maps[k] for k in self._policy._cat])
        mode = AlignmentMode(mode)  # type: ignore[assignment]
        if mode == "affine":
            if not inplace:
                return aligned_basis, aligned_metadata
            if self.spatial_key not in self.adata.uns:
                self.adata.uns[self.spatial_key] = {}
            self.adata.uns[self.spatial_key]["alignment_metadata"] = aligned_metadata
        if not inplace:
            return aligned_basis
        self.adata.obsm[f"{self.spatial_key}_{mode}"] = aligned_basis

    def _subset_spatial(self: SpatialAlignmentMixinProtocol[K, B], k: K) -> ArrayLike:
        return self.adata[self.adata.obs[self._policy._subset_key] == k].obsm[self.spatial_key].copy()

    def cell_transition(
        self: SpatialAlignmentMixinProtocol[K, B],
        slice: K,
        reference: K,
        slice_cells: Filter_t,
        reference_cells: Filter_t,
        forward: bool = False,  # return value will be row-stochastic if forward=True, else column-stochastic
        aggregation_mode: Literal["annotation", "cell"] = AggregationMode.ANNOTATION,  # type: ignore[assignment]
        online: bool = False,
        batch_size: Optional[int] = None,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """Partly copy from other cell_transitions."""
        if TYPE_CHECKING:
            assert isinstance(self.batch_key, str)
        return self._cell_transition(
            key=self.batch_key,
            source_key=slice,
            target_key=reference,
            source_annotation=slice_cells,
            target_annotation=reference_cells,
            forward=forward,
            aggregation_mode=AggregationMode(aggregation_mode),
            online=online,
            other_key=None,
            other_adata=None,
            batch_size=batch_size,
            normalize=normalize,
        )

    @property
    def spatial_key(self) -> Optional[str]:
        """Return spatial key."""
        return self._spatial_key

    @spatial_key.setter
    def spatial_key(self: SpatialAlignmentMixinProtocol[K, B], value: Optional[str]) -> None:
        if value is not None and value not in self.adata.obsm:
            raise KeyError(f"TODO: {value} not found in `adata.obsm`.")
        # TODO(@MUCDK) check data type -> which ones do we allow
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
        corr_method: Literal["pearson", "spearman"] = CorrMethod.PEARSON,  # type: ignore[assignment]
    ) -> Mapping[Tuple[K, K], pd.Series]:
        """
        Calculate correlation between true and predicted gene expression.

        Parameters
        ----------
        var_names
            List of variable names.
        corr_method
            Correlation method:
            - "pearson": Pearson correlation.
            - "spearman": Spearman correlation

        Returns
        -------
        :class:`pandas.DataFrame` with correlation results.
        """
        var_sc = self._filter_vars(var_names)
        if var_sc is None or not len(var_sc):
            raise ValueError("No overlapping `var_names` between ` adata_sc` and `adata_sp`.")
        corr_method = CorrMethod(corr_method)  # type: ignore[assignment]
        if corr_method == CorrMethod.PEARSON:  # type: ignore[comparison-overlap]
            cor = pearsonr
        elif corr_method == CorrMethod.SPEARMAN:  # type: ignore[comparison-overlap]
            cor = spearmanr
        else:
            raise NotImplementedError("TODO: `corr_method` must be `pearson` or `spearman`.")
        corr_dic = {}
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
            corr_dic[key] = pd.Series(corr_val, index=var_sc)

        return corr_dic

    def impute(self: SpatialMappingMixinProtocol[K, B]) -> AnnData:
        """
        Impute expression of specific genes.

        Parameters
        ----------
        None

        Returns
        -------
        :class:`anndata.AnnData` with imputed gene expression values.
        """
        gexp_sc = self.adata_sc.X if not issparse(self.adata_sc.X) else self.adata_sc.X.A
        pred_list = [val.pull(gexp_sc, scale_by_marginals=True) for val in self.solutions.values()]
        adata_pred = AnnData(np.nan_to_num(np.vstack(pred_list), nan=0.0, copy=False))
        adata_pred.obs_names = self.adata_sp.obs_names.copy()
        adata_pred.var_names = self.adata_sc.var_names.copy()
        return adata_pred

    def cell_transition(
        self: SpatialMappingMixinProtocol[K, B],
        batch: K,
        spatial_cells: Filter_t,
        sc_cells: Filter_t,
        forward: bool = False,  # return value will be row-stochastic if forward=True, else column-stochastic
        aggregation_mode: Literal["annotation", "cell"] = AggregationMode.ANNOTATION,  # type: ignore[assignment]
        online: bool = False,
        batch_size: Optional[int] = None,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """Compute cell transition."""
        if TYPE_CHECKING:
            assert self.batch_key is not None
        return self._cell_transition(
            key=self.batch_key,
            source_key=batch,
            target_key=None,
            source_annotation=spatial_cells,  # change it to source_groups
            target_annotation=sc_cells,  # change it to target_groups
            forward=forward,
            aggregation_mode=AggregationMode(aggregation_mode),
            online=online,
            other_key=None,
            other_adata=self.adata_sc,
            batch_size=batch_size,
            normalize=normalize,
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
