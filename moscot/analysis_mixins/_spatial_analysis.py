from typing import Any, Dict, List, Tuple, Union, Mapping, Optional, Sequence

from scipy.stats import pearsonr, spearmanr
from scipy.linalg import svd
from scipy.sparse import issparse
from typing_extensions import Literal
from scipy.sparse.linalg import LinearOperator
import pandas as pd

import numpy as np
import numpy.typing as npt

from anndata import AnnData

from moscot.problems._subset_policy import StarPolicy
from moscot.analysis_mixins._base_analysis import AnalysisMixin


class SpatialAlignmentAnalysisMixin(AnalysisMixin):
    """Spatial alignment mixin class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._spatial_key: Optional[str] = None

    def _interpolate_scheme(self, reference: Any, mode: Literal["warp", "affine"]) -> Dict[str, npt.ArrayLike]:
        """Scheme for interpolation."""

        def subs_adata(k: Any) -> npt.ArrayLike:
            return self.adata[self.adata.obs[self._policy._subset_key] == k].obsm[self.spatial_key].copy()

        # TODO: error message for star policy
        # get reference
        src = subs_adata(reference)
        transport_maps = {reference: src}
        transport_metadata = {}
        if mode == "affine":
            src -= src.mean(0)
            transport_metadata = {reference: 0}

        # get policy
        full_steps = self._policy._subset
        fwd_steps = self._policy.plan(end=reference)
        bwd_steps = None
        # get mapping function
        _transport = self._affine if mode == "affine" else lambda tmap, _, src: (tmap.dot(src), None)

        if not fwd_steps or not set(full_steps).issubset(set(fwd_steps.keys())):
            bwd_steps = self._policy.plan(start=reference)

        if len(fwd_steps):
            for (start, end) in fwd_steps.keys():
                tmap = self._interpolate_transport(start=start, end=end, scale_by_marginals=True, forward=True)
                transport_maps[start], transport_metadata[start] = _transport(tmap, subs_adata(start), src)

        if bwd_steps is not None and len(bwd_steps):
            for (start, end) in bwd_steps.keys():
                tmap = self._interpolate_transport(start=start, end=end, scale_by_marginals=True, forward=False)
                transport_maps[end], transport_metadata[end] = _transport(tmap.T, subs_adata(end), src)

        if mode == "affine":
            return transport_maps, transport_metadata
        return transport_maps, None

    def _affine(
        self, tmap: Union[npt.ArrayLike, LinearOperator], tgt: npt.ArrayLike, src: npt.ArrayLike
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        """Affine transformation."""
        tgt -= tgt.mean(0)
        H = tgt.T.dot(tmap.dot(src))
        U, _, Vt = svd(H)
        R = Vt.T.dot(U.T)
        tgt = R.dot(tgt.T).T
        return tgt, R

    def align(
        self,
        reference: Any,
        mode: Literal["warp", "affine"] = "warp",
        inplace: bool = False,
    ) -> Optional[Union[npt.ArrayLike, Tuple[npt.ArrayLike, Dict[Any, npt.ArrayLike]]]]:
        """Alignemnt method."""
        if reference not in self._policy._cat.categories:
            raise ValueError(f"`reference: {reference}` not in policy categories: {self._policy._cat.categories}.")
        if isinstance(self._policy, StarPolicy):
            if reference != list(self._policy.plan().keys())[0][-1]:
                raise ValueError(f"Invalid `reference: {reference}` for `policy='star'`.")
        aligned_maps, aligned_metadata = self._interpolate_scheme(reference=reference, mode=mode)
        aligned_basis = np.vstack([aligned_maps[k] for k in self._policy._cat.categories])

        if mode == "affine":
            if inplace:
                return aligned_basis, aligned_metadata
            self.adata.uns[self.spatial_key]["alignment_metadata"] = aligned_metadata
        if inplace:
            return aligned_basis
        self.adata.obsm[f"{self.spatial_key}_{mode}"] = aligned_basis

    @property
    def spatial_key(self) -> Optional[str]:
        """Return spatial key."""
        return self._spatial_key

    @spatial_key.setter
    def spatial_key(self, value: Optional[str] = None) -> None:
        if value not in self.adata.obsm:
            raise KeyError(f"TODO: {value} not found in `adata.obsm`.")
        # TODO(@MUCDK) check data type -> which ones do we allow
        self._spatial_key = value


class SpatialMappingAnalysisMixin(AnalysisMixin):
    """Spatial mapping analysis mixin class."""

    def _filter_vars(
        self,
        var_names: Optional[Sequence[Any]] = None,
    ) -> Optional[List[str]]:
        """Filter variables for Sinkhorn tem."""
        vars_sc = set(self.adata_sc.var_names)  # TODO: allow alternative gene symbol by passing var_key
        vars_sp = set(self.adata.var_names)
        var_names = set(var_names) if var_names is not None else None
        if var_names is None:
            var_names = vars_sp.intersection(vars_sc)
            if len(var_names):
                return list(var_names)
            raise ValueError("`adata_sc` and `adata_sp` do not share `var_names`. Input valid `var_names`.")
        if not len(var_names):
            return None
        if var_names.issubset(vars_sc) and var_names.issubset(vars_sp):
            return list(var_names)
        raise ValueError("Some `var_names` ares missing in either `adata_sc` or `adata_sp`.")

    def correlate(
        self, var_names: Optional[List[str]] = None, corr_method: Literal["pearson", "spearman"] = "pearson"
    ) -> Mapping[Tuple[str, Any], pd.Series]:
        """Calculate correlation between true and predicted gexp in space."""
        var_sc = self._filter_vars(var_names)
        if not len(var_sc):
            raise ValueError("No overlapping `var_names` between ` adata_sc` and `adata_sp`.")
        cor = pearsonr if corr_method == "pearson" else spearmanr
        corr_dic = {}
        gexp_sc = self.adata_sc[:, var_sc].X if not issparse(self.adata_sc.X) else self.adata_sc[:, var_sc].X.A
        for prob_key, prob_val in self.solutions.items():
            index_obs: List[Union[bool, int]] = (
                self.adata.obs[self._policy._subset_key] == prob_key[0]
                if self._policy._subset_key is not None
                else np.arange(self.adata_sp.shape[0])
            )
            gexp_sp = (
                self.adata[index_obs, var_sc].X if not issparse(self.adata.X) else self.adata[index_obs, var_sc].X.A
            )
            gexp_pred_sp = prob_val.pull(gexp_sc, scale_by_marginals=True)
            corr_val = [cor(gexp_pred_sp[:, gi], gexp_sp[:, gi])[0] for gi, _ in enumerate(var_sc)]
            corr_dic[prob_key] = pd.Series(corr_val, index=var_sc)

        return corr_dic

    def impute(self) -> AnnData:
        """Return imputation of spatial expression of given genes."""
        gexp_sc = self.adata_sc.X if not issparse(self.adata_sc.X) else self.adata_sc.X.A
        pred_list = []
        for prob_val in self.solutions.values():
            pred_list.append(prob_val.pull(gexp_sc, scale_by_marginals=True))
        adata_pred = AnnData(np.nan_to_num(np.vstack(pred_list), nan=0.0, copy=False))
        adata_pred.obs_names = self.adata.obs_names.values.copy()
        adata_pred.var_names = self.adata_sc.var_names.values.copy()
        return adata_pred
