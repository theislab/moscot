from __future__ import annotations

from typing import Any, List, Tuple, Mapping, Optional, NamedTuple

from scipy.stats import pearsonr, spearmanr
from scipy.sparse import issparse
from typing_extensions import Literal
import pandas as pd

import numpy as np
import numpy.typing as npt

from anndata import AnnData

from moscot.mixins._base_analysis import AnalysisMixin


class AlignParams(NamedTuple):
    """Class to store alignment paramss."""

    mode: str | None
    tmap: npt.ArrayLike | None
    basis: npt.ArrayLike


class SpatialAnalysisMixin(AnalysisMixin):
    """Spatial analysis mixin class."""

    def _interpolate_transport(
        self, start: int | str, end: int | str, forward: bool = True, normalize: bool = True
    ) -> npt.ArrayLike:
        """Interpolate transport matrix."""
        steps = self._policy.plan(start=start, end=end)[start, end]
        transition_matrix = self._problems[steps[0]].solution._scale_transport_by_marginals(forward=True)
        if len(steps) == 1:
            return transition_matrix
        for i in range(len(steps) - 1):
            transition_matrix = transition_matrix @ self._problems[steps[i + 1]].solution._scale_transport_by_marginals(
                forward=True
            )
        if normalize:
            if forward:
                return transition_matrix / transition_matrix.sum(0)[None, :]
            else:
                return transition_matrix / transition_matrix.sum(1)[:, None]
        return transition_matrix


class SpatialAlignmentAnalysisMixin(SpatialAnalysisMixin):
    """Spatial alignment mixin class."""

    def _interpolate_scheme(self, reference: str | int) -> Mapping[Tuple[Any, Any] | Tuple[Any], npt.ArrayLike]:
        """Scheme for interpolation."""
        full_steps = self._policy._subset
        fwd_steps = self._policy.plan(end=reference)
        bwd_steps = None
        if not fwd_steps or not set(full_steps).issubset(set(fwd_steps.keys())):
            bwd_steps = self._policy.plan(start=reference)
        dic_transport = {}

        def subs_adata(k: str | int) -> npt.ArrayLike:
            return self.adata[self.adata.obs[self._policy._subset_key] == k].obsm[self.spatial_key]

        if len(fwd_steps):
            for k in fwd_steps.keys():
                start, end = k
                tmap = self._interpolate_transport(start=start, end=end, forward=False)  # fwd arg for norm
                dic_transport[k] = AlignParams("fwd", tmap, subs_adata(end))

        dic_transport[(reference, None)] = AlignParams(None, None, subs_adata(reference))
        if bwd_steps is not None and len(bwd_steps):
            for k in bwd_steps.keys():
                start, end = k
                tmap = self._interpolate_transport(start=start, end=end, forward=True)
                dic_transport[k] = AlignParams("bwd", tmap, subs_adata(start))
        return dic_transport

    def _warp(basis: npt.ArrayLike, tmap: npt.ArrayLike):

        return

    def align(
        self,
        reference: str | int,
        key: str = "spatial_warp",
        mode: Literal["warp", "affine"] = "warp",
        copy: bool = False,
    ) -> npt.ArrayLike | None:
        """Spatial warp."""
        if reference not in self._policy._cat.categories:
            raise ValueError(f"`reference: {reference}` not in policy categories: {self._policy._cat.categories}")
        dic_transport = self._interpolate_scheme(reference=reference)

        basis_list = []
        for _, v in dic_transport.items():
            if v.mode is not None:
                if v.mode == "fwd":
                    basis_list.append((v.basis.T @ v.tmap.T).T)
                elif v.mode == "bwd":
                    basis_list.append((v.basis.T @ v.tmap).T)
            else:
                basis_list.append(v.basis)

        aligned_arr = np.vstack(basis_list)

        if copy:
            return aligned_arr

        self.adata.obsm[key] = aligned_arr

    def align_affine(self, reference: str | int, key: str = "spatial_warp", copy: bool = False) -> npt.ArrayLike | None:
        """Spatial warp."""
        if reference not in self._policy._cat.categories:
            raise ValueError(f"`reference: {reference}` not in policy categories: {self._policy._cat.categories}")
        dic_transport = self._interpolate_scheme(reference=reference)

        basis_list = []
        for _, v in dic_transport.items():
            if v.mode is not None:
                if v.mode == "fwd":
                    basis_list.append((v.basis.T @ v.tmap).T)
                elif v.mode == "bwd":
                    basis_list.append((v.basis.T @ v.tmap.T).T)
            else:
                basis_list.append(v.basis)

        aligned_arr = np.vstack(basis_list)

        if copy:
            return aligned_arr

        self.adata.obsm[key] = aligned_arr


class SpatialMappingAnalysisMixin(SpatialAnalysisMixin):
    """Spatial mapping analysis mixin class."""

    def _filter_vars(
        self,
        adata_sc: AnnData,
        adata_sp: AnnData,
        var_names: Optional[List[str]] = None,
        use_reference: bool = False,
    ) -> List[str] | None:
        """Filter variables for Sinkhorn tem."""
        vars_sc = set(adata_sc.var_names)  # TODO: allow alternative gene symbol by passing var_key
        vars_sp = set(adata_sp.var_names)
        var_names = set(var_names) if var_names is not None else None
        if use_reference is True and var_names is None:
            var_names = vars_sp.intersection(vars_sc)
            if len(var_names):
                return list(var_names)
            raise ValueError("`adata_sc` and `adata_sp` do not share `var_names`. Input valid `var_names`.")
        if not use_reference:
            return None
        if use_reference and var_names.issubset(vars_sc) and var_names.issubset(vars_sp):
            return list(var_names)
        raise ValueError("Some `var_names` ares missing in either `adata_sc` or `adata_sp`.")

    def correlate(
        self, var_names: List[str] | None = None, corr_method: Literal["pearson", "spearman"] = "pearson"
    ) -> Mapping[Tuple[str, Any], pd.Series]:
        """Calculate correlation between true and predicted gexp in space."""
        var_sc = self._filter_vars(self.adata_sc, self.adata_sp, var_names, True)
        if not len(var_sc):
            raise ValueError("No overlapping `var_names` between ` adata_sc` and `adata_sp`.")
        cor = pearsonr if corr_method == "pearson" else spearmanr
        corr_dic = {}
        gexp_sc = self.adata_sc[:, var_sc].X if not issparse(self.adata_sc.X) else self.adata_sc[:, var_sc].X.A
        for prob_key, prob_val in self.solution.items():
            index_obs: List[bool | int] = (
                self.adata_sp.obs[self._policy._subset_key] == prob_key[0]
                if self._policy._subset_key is not None
                else np.arange(self.adata_sp.shape[0])
            )
            gexp_sp = (
                self.adata_sp[index_obs, var_sc].X
                if not issparse(self.adata_sp.X)
                else self.adata_sp[index_obs, var_sc].X.A
            )
            tmap = prob_val.solution._scale_transport_by_marginals(forward=False)
            gexp_pred_sp = np.dot(tmap, gexp_sc)
            corr_val = [cor(gexp_pred_sp[:, gi], gexp_sp[:, gi])[0] for gi, _ in enumerate(var_sc)]
            corr_dic[prob_key] = pd.Series(corr_val, index=var_sc)

        return corr_dic

    def impute(self) -> AnnData:
        """Return imputation of spatial expression of given genes."""
        gexp_sc = self.adata_sc.X if not issparse(self.adata_sc.X) else self.adata_sc.X.A
        pred_list = []
        for _, prob_val in self.solution.items():
            tmap = prob_val.solution._scale_transport_by_marginals(forward=False)
            pred_list.append(np.dot(tmap, gexp_sc))
        adata_pred = AnnData(np.nan_to_num(np.vstack(pred_list), nan=0.0, copy=False))
        adata_pred.obs_names = self.adata_sp.obs_names.values.copy()
        adata_pred.var_names = self.adata_sc.var_names.values.copy()
        return adata_pred
