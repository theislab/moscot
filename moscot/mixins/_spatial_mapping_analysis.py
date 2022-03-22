from moscot.mixins._spatial_analysis import SpatialAnalysisMixin
from moscot.problems.spatial import SpatialMappingProblem
from typing import Any, Dict, Tuple, Union, Optional, Iterable, Mapping

import numpy.typing as npt
import jax.numpy as jnp
import numpy as np
import pandas as pd

import scipy
from scipy.stats import pearsonr

class SpatialMappingAnalysisMixin(SpatialAnalysisMixin):


    def filter_vars(
        self,
        adata_sc: AnnData,
        adata_sp: Optional[AnnData] = None,
        var_names: Optional[List[str]] = None,
    ) -> Tuple[AnnData, AnnData]:
        vars_sc = set(adata_sc.var_names)  # TODO: allow alternative gene symbol by passing var_key
        vars_sp = set(adata_sp.var_names) if adata_sp is not None
        var_names = set(var_names) if var_names is not None else None
        if var_names is None and adata_sp is not None:
            var_names = vars_sp.intersection(vars_sc)
            if len(var_names):
                return adata_sc[:, list(var_names)], adata_sp[:, list(var_names)]
            else:
                logg.warning(f"`adata_sc` and `adata_sp` do not share `var_names`. ")
                return adata_sc, adata_sp
        elif var_names is None:
            return adata_sc
        elif adata_sp is not None:
            if var_names.issubset(vars_sc) and var_names.issubset(vars_sp):
                return adata_sc[:, list(var_names)], adata_sp[:, list(var_names)]
            else:
                raise ValueError("Some `var_names` ares missing in either `adata_sc` or `adata_sp`.")
        else:
            if var_names.issubset(vars_sc):
                return adata_sc[:, list(var_names)]
            else:
                raise ValueError("Some `var_names` ares missing in `adata_sc`.")


    def correlate(self,
                   adata_sc,
                   adata_sp,
                   transport_matrix: npt.ArrayLike,
                   var_names: Optional[List[str]] = None,
                   key_pred: str = None):
        """
        calculate correlation between the predicted gene expression and observed in tissue space.
        Parameters
        ----------
        transport_matrix: learnt transport_matrix - - assumes [n_cell_sp X n_cells_sc]
        var_names: genes to correlate, if none correlate all.
        Returns
        -------
        corr_val: the pearsonr correlation
        """
        adata_sc, adata_sp = self.filter_vars(adata_sc, adata_sp, var_names)
        if scipy.sparse.issparse(adata_sc.X):
            adata_sc.X = adata_sc.X.A
        if scipy.sparse.issparse(adata_sp.X):
            adata_sp.X = adata_sp.X.A
        sp_gex_pred = np.asarray(jnp.dot(transport_matrix, adata_sc.X))
        corr_val = np.nanmean([pearsonr(sp_gex_pred[:, gi],
                                        adata_sp.X[:, gi])[0]
                               for gi, g in enumerate(adata_sp.var_names)])
        return corr_val

    def get_imputation(self,
                       adata_sc,
                       adata_sp,
                       transport_matrix: npt.ArrayLike,
                       var_names: Optional[List[str]] = None,) -> pd.DataFrame:
        """
        return imputation of spatial expression of given genes
        Parameters
        ----------
        transport_matrix: learnt transport_matrix - - assumes [n_cell_sp X n_cells_sc]
        var_names: genes to correlate, if none correlate all.
        Returns
        -------
        genes_impute; df of spatial gex
        """
        adata_sc = self.filter_vars(adata_sc, var_names=var_names)
        if scipy.sparse.issparse(adata_sc.X):
            adata_sc.X = adata_sc.X.A
        sp_gex_pred = np.asarray(jnp.dot(transport_matrix, adata_ref.X))
        sp_gex_pred = pd.DataFrame(sp_gex_pred,
                                   index=adata_sp.obs_names,
                                   columns=adata_sc.var_names)
        return sp_gex_pred

