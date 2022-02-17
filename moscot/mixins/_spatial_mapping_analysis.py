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

    def __init__(self, sols: SpatialMappingProblem):
        self._sols = sols
        self._adata_sc = sols.adata_sc
        self._adata_sp = sols.adata_sp
        self._corr_vals = None

    def _correlate(self, transport_matrix: npt.ArrayLike,
                   mask_sc: npt.ArrayLike,
                   key_pred: str = None):
        """
        calculate correlation between the predicted gene expression and observed in tissue space.
        Parameters
        ----------
        transport_matrix: learnt transport_matrix
        mask_sc: mask which indicates marker genes to use for sc data.
        key_pred: if provided spatial gex prediction will be added as a spatial obsm using this key
        Returns
        -------
        corr_val: the pearsonr correlation
        """
        adata_ref = self._adata_sc[:, mask_sc]
        adata_sp_ref = self._adata_sp[:, adata_ref.var['SYMBOL'].astype(str).values]
        if scipy.sparse.issparse(adata_ref.X):
            adata_ref.X = adata_ref.X.A
        if scipy.sparse.issparse(adata_sp_ref.X):
            adata_sp_ref.X = adata_sp_ref.X.A
        sp_gex_pred = np.asarray(jnp.dot(adata_ref.X.T, transport_matrix).T)
        corr_val = np.nanmean([pearsonr(sp_gex_pred[:, gi],
                                        adata_sp_ref.X[:, gi])[0]
                               for gi, g in enumerate(adata_sp_ref.var_names)])
        if key_pred is not None:
            self._adata_sp.obsm[key_pred] = pd.DataFrame(sp_gex_pred,
                                                         index=self._adata_sp.obs_names,
                                                         columns=adata_sp_ref.var_names)
        return corr_val

    def get_imputation(self, keys_subset: Optional[Union[str, Dict[Any, Tuple[str, str]]]] = None,
                       var_subset: Optional[Union[Tuple[Any, Any], Dict[Any, Tuple[Any, Any]]]] = None):
        """
        return imputation of spatial expression of given genes
        Parameters
        ----------
        keys_subset: key(s) for .var which indicate marker genes.
            either a single key or a key for each problem.
        var_subset: subset(s) of marker genes to use, either a single list or dictionary of lists.
            either a single list or a lists for each problem.
        Returns
        -------
        genes_impute; df of spatial gex
        """
        for prob_key, prob_val in self._sols.problems.items():
            if var_subset is not None:
                if isinstance(var_subset, Mapping):
                    mask = var_subset[prob_key]
                else:
                    mask = var_subset
            elif keys_subset is not None:
                if isinstance(keys_subset, Mapping):
                    mask = prob_val.adata.var[keys_subset[prob_key]]
                else:
                    mask = prob_val.adata.var[keys_subset]
            else:
                print("no genes for comparison were provided")
                return
            adata_ref = self._adata_sc[:, mask]
            adata_sp_ref = self._adata_sp[:, adata_ref.var['SYMBOL'].astype(str).values]
            if scipy.sparse.issparse(adata_ref.X):
                adata_ref.X = adata_ref.X.A
            if scipy.sparse.issparse(adata_sp_ref.X):
                adata_sp_ref.X = adata_sp_ref.X.A

            sp_gex_pred = np.asarray(jnp.dot(adata_ref.X.T, prob_val.solution.scaled_transport(forward=True)).T)
            sp_gex_pred = pd.DataFrame(sp_gex_pred,
                                       index=self._adata_sp.obs_names,
                                       columns=adata_sp_ref.var_names)
            return sp_gex_pred

    def correlate(self, keys_subset: Optional[Union[str, Dict[Any, Tuple[str, str]]]] = None,
                  var_subset: Optional[Union[Tuple[Any, Any], Dict[Any, Tuple[Any, Any]]]] = None,
                  key_pred: Optional[Union[str, Dict[Any, Tuple[str, str]]]] = None) -> "SpatialMappingAnalysisMixin":
        """
        compute correlation of spatial mappings sols wrt given genes
        Parameters
        ----------
        keys_subset: key(s) for .var which indicate marker genes (expects identical keys for `spatial` and 'scRNA' adata).
         either a single key or a key for each problem.
        var_subset: subset(s) of marker genes to use, either a single list or dictionary of lists.
        either a single list or a lists for each problem.
        key_pred: if provided spatial gex prediction will be added as a spatial obsm using this key
        Returns
        -------
        Saves a dict of correlation values
        """
        self._corr_vals = {}
        for prob_key, prob_val in self._sols.problems.items():
            if var_subset is not None:
                if isinstance(var_subset, Mapping):
                    mask = var_subset[prob_key]
                else:
                    mask = var_subset
            elif keys_subset is not None:
                if isinstance(keys_subset, Mapping):
                    mask = prob_val.adata.var[keys_subset[prob_key]]
                else:
                    mask = prob_val.adata.var[keys_subset]
            else:
                print("no genes for comparison were provided")
                return
            key_pred_ = None
            if key_pred is not None:
                if isinstance(keys_subset, Mapping):
                    key_pred_ = key_pred[prob_key]
                else:
                    key_pred_ = '_'.join((prob_key, key_pred))
            self._corr_vals[prob_key] = self._correlate(prob_val.solution.scaled_transport(forward=True),
                                                        mask_sc=mask,
                                                        key_pred=key_pred_)
            return self