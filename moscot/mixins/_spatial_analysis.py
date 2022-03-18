from __future__ import annotations

from typing import Set, List, Optional

from scipy.stats import pearsonr, spearmanr
from scipy.sparse import issparse
from typing_extensions import Literal
import pandas as pd

import numpy as np

from anndata import AnnData

from moscot.mixins._base_analysis import AnalysisMixin


class SpatialAnalysisMixin(AnalysisMixin):
    """Spatial analysis mixin class."""


class SpatialAlignmentAnalysisMixin(AnalysisMixin):
    """Spatial alignment mixin class."""


class SpatialMappingAnalysisMixin(SpatialAnalysisMixin):
    """Spatial mapping analysis mixin class."""

    def _filter_vars(
        self,
        adata_sc: AnnData,
        adata_sp: AnnData,
        var_names: Optional[List[str]] = None,
        use_reference: bool = False,
    ) -> Set | None:
        """Filter variables for Sinkhorn tem."""
        vars_sc = set(adata_sc.var_names)  # TODO: allow alternative gene symbol by passing var_key
        vars_sp = set(adata_sp.var_names)
        var_names = set(var_names) if var_names is not None else None
        if use_reference is True and var_names is None:
            var_names = vars_sp.intersection(vars_sc)
            if len(var_names):
                return var_names
            raise ValueError("`adata_sc` and `adata_sp` do not share `var_names`. Input valid `var_names`.")
        if not use_reference:
            return None
        if use_reference and var_names.issubset(vars_sc) and var_names.issubset(vars_sp):
            return var_names
        raise ValueError("Some `var_names` ares missing in either `adata_sc` or `adata_sp`.")

    def correlate(self, var_names: List[str] | None = None, corr_method: Literal["pearson", "spearman"] = "pearson"):
        """Calculate correlation between true and predicted gexp in space."""
        var_sc = self._filter_vars(self.adata_sc, self.adata_sp, var_names, True)

        var_sc = list(set(self.adata_sc.var_names).intersection(self.adata_sp.var_names))
        if not len(var_sc):
            raise ValueError("No overlapping `var_names` between ` adata_sc` and `adata_sp`.")
        cor = pearsonr if corr_method == "pearson" else spearmanr
        corr_dic = {}
        gexp_sc = self.adata_sc[:, var_sc].X if not issparse(self.adata_sc.X) else self.adata_sc[:, var_sc].X.A
        for prob_key, prob_val in self.solution.items():
            index_obs = self.adata_sp.obs[self._policy._subset_key] == prob_key[0]
            gexp_sp = (
                self.adata_sp[index_obs, var_sc].X
                if not issparse(self.adata_sp.X)
                else self.adata_sp[index_obs, var_sc].X.A
            )
            transport_matrix = prob_val.solution.scaled_transport(forward=False)
            gexp_pred_sp = np.dot(transport_matrix, gexp_sc)
            corr_val = [cor(gexp_pred_sp[:, gi], gexp_sp[:, gi])[0] for gi, _ in enumerate(var_sc)]
            corr_dic[prob_key] = pd.Series(corr_val, index=var_sc)

        return corr_dic

    def impute(self) -> AnnData:
        """Return imputation of spatial expression of given genes."""
        gexp_sc = self.adata_sc.X if not issparse(self.adata_sc.X) else self.adata_sc.X.A
        pred_list = []
        for _, prob_val in self.solution.items():
            transport_matrix = prob_val.solution.scaled_transport(forward=False)
            pred_list.append(np.dot(transport_matrix, gexp_sc))
        adata_pred = AnnData(np.nan_to_num(np.vstack(pred_list), nan=0.0, copy=False))
        adata_pred.obs_names = self.adata_sp.obs_names.values.copy()
        adata_pred.var_names = self.adata_sc.var_names.values.copy()
        return adata_pred
