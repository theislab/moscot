from typing import Any, Dict, List, Tuple, Union, Mapping, Optional, Sequence

from scipy.stats import pearsonr, spearmanr
from scipy.linalg import svd
from scipy.sparse import issparse
from typing_extensions import Literal
import pandas as pd

import numpy as np
import numpy.typing as npt

from anndata import AnnData

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
        if mode == "affine":
            src -= src.mean(0)
        dic_transport = {reference: src}
        # get policy
        full_steps = self._policy._subset
        fwd_steps = self._policy.plan(end=reference)
        bwd_steps = None
        # get mapping function
        fun_transport = self._affine if mode == "affine" else lambda tmap, _, src: tmap @ src

        if not fwd_steps or not set(full_steps).issubset(set(fwd_steps.keys())):
            bwd_steps = self._policy.plan(start=reference)

        if len(fwd_steps):
            for (start, end) in fwd_steps.keys():
                tmap = self._interpolate_transport(start=start, end=end, normalize=True, forward=True)
                dic_transport[start] = fun_transport(tmap, subs_adata(start), src)

        if bwd_steps is not None and len(bwd_steps):
            for (start, end) in bwd_steps.keys():
                tmap = self._interpolate_transport(start=start, end=end, normalize=True, forward=False)
                dic_transport[end] = fun_transport(tmap.T, subs_adata(end), src)

        return dic_transport

    def _affine(self, tmap: npt.ArrayLike, tgt: npt.ArrayLike, src: npt.ArrayLike) -> npt.ArrayLike:
        """Affine transformation."""
        tgt -= tgt.mean(0)
        H = tgt.T.dot(tmap.dot(src))
        U, _, Vt = svd(H)
        R = Vt.T.dot(U.T)
        tgt = R.dot(tgt.T).T
        return tgt

    def align(
        self,
        reference: Any,
        mode: Literal["warp", "affine"] = "warp",
        copy: bool = False,
    ) -> Optional[npt.ArrayLike]:
        """Spatial warp."""
        if reference not in self._policy._cat.categories:
            raise ValueError(f"`reference: {reference}` not in policy categories: {self._policy._cat.categories}")
        aligned_dic = self._interpolate_scheme(reference=reference, mode=mode)
        aligned_arr = np.vstack([aligned_dic[k] for k in self._policy._cat.categories])

        if copy:
            return aligned_arr

        self.adata.obsm[f"{self.spatial_key}_{mode}"] = aligned_arr

    @property
    def spatial_key(self) -> Optional[str]:
        """Return spatial key."""
        return self._spatial_key

    @spatial_key.setter
    def spatial_key(self, value: Optional[str] = None) -> None:
        if value not in self.adata.obs.columns:
            raise KeyError(f"TODO: {value} not found in `adata.obs.columns`")
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
        var_sc = self._filter_vars(var_names, True)
        if not len(var_sc):
            raise ValueError("No overlapping `var_names` between ` adata_sc` and `adata_sp`.")
        cor = pearsonr if corr_method == "pearson" else spearmanr
        corr_dic = {}
        gexp_sc = self.adata_sc[:, var_sc].X if not issparse(self.adata_sc.X) else self.adata_sc[:, var_sc].X.A
        for prob_key, prob_val in self.solution.items():
            index_obs: List[Union[bool, int]] = (
                self.adata.obs[self._policy._subset_key] == prob_key[0]
                if self._policy._subset_key is not None
                else np.arange(self.adata_sp.shape[0])
            )
            gexp_sp = (
                self.adata[index_obs, var_sc].X if not issparse(self.adata.X) else self.adata[index_obs, var_sc].X.A
            )
            tmap = prob_val._scale_transport_by_marginals(forward=False)
            gexp_pred_sp = np.dot(tmap, gexp_sc)
            corr_val = [cor(gexp_pred_sp[:, gi], gexp_sp[:, gi])[0] for gi, _ in enumerate(var_sc)]
            corr_dic[prob_key] = pd.Series(corr_val, index=var_sc)

        return corr_dic

    def impute(self) -> AnnData:
        """Return imputation of spatial expression of given genes."""
        gexp_sc = self.adata_sc.X if not issparse(self.adata_sc.X) else self.adata_sc.X.A
        pred_list = []
        for _, prob_val in self.solution.items():
            tmap = prob_val._scale_transport_by_marginals(forward=False)
            pred_list.append(np.dot(tmap, gexp_sc))
        adata_pred = AnnData(np.nan_to_num(np.vstack(pred_list), nan=0.0, copy=False))
        adata_pred.obs_names = self.adata.obs_names.values.copy()
        adata_pred.var_names = self.adata_sc.var_names.values.copy()
        return adata_pred
