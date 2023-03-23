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
from anndata import AnnData
import scipy.spatial
from sklearn.preprocessing import normalize
import anndata as ad
import scanpy as sc


from moscot import _constants
from moscot._docs._docs import d
from moscot._docs._docs_mixins import d_mixins
from moscot._logging import logger
from moscot._types import ArrayLike, Device_t, Str_Dict_t
from moscot.base.problems._mixins import AnalysisMixin, AnalysisMixinProtocol
from moscot.base.problems.compound_problem import B, K
from moscot.utils.subset_policy import StarPolicy
from moscot.base.output import BaseSolverOutput


__all__ = ["CrossModalityIntegrationMixin"]

class CrossModalityIntegrationMixinProtocol(AnalysisMixinProtocol[K, B]):
    """Protocol class."""

    adata_src: AnnData
    adata_tgt: AnnData
    src_attr: Optional[str]
    tgt_attr: Optional[str]

    def _normalize(
            self: AnalysisMixinProtocol[K, B],##this or  CrossModalityIntegrationMixinProtocol
            norm="l2", 
            bySample=True
    ):
        ...

    def _barycentric_projection(
            self: AnalysisMixinProtocol[K, B],
            SRContoTGT=True
    ):
        ...   

    def solution(
        self: AnalysisMixinProtocol[K, B],
    ) -> Optional[BaseSolverOutput]:
        ... 

class CrossModalityIntegrationMixin(AnalysisMixin[K, B]):
    """Cross modality integration analysis mixin class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._src_attr: Optional[str] = None
        self._tgt_attr: Optional[str] = None

    def _normalize(
            self: CrossModalityIntegrationMixinProtocol[K, B],
            norm="l2", 
            bySample=True
    ):
        """
        Determines what sort of normalization to run, "l2", "l1", "max". Default="l2" 

        Parameters
        ----------
        %(norm)s
        %(bySample)s

        Returns
        -------
        
        """
        assert (norm in ["l1","l2","max"]), "Norm argument has to be either one of 'max', 'l1', or 'l2'."
        if (bySample==True or bySample==None):
            axis=1
        else:
            axis=0
        self.adata_src.obsm[self._src_attr] = normalize(self.adata_src.obsm[self._src_attr], norm=norm, axis=axis)
        self.adata_tgt.obsm[self._tgt_attr] = normalize(self.adata_tgt.obsm[self._tgt_attr], norm=norm, axis=axis)

    def _barycentric_projection(
            self: CrossModalityIntegrationMixinProtocol[K, B],
            SRContoTGT=True
    ):
        """
        Determines the direction of barycentric projection. True or False (boolean parameter). 
        If True, projects domain1 onto domain2. 
        If False, projects domain2 onto domain1. 
        Default=True.

        Parameters
        ----------
        %(SRContoTGT)s

        Returns
        -------
        
        """

        if SRContoTGT:
            # Projecting the source domain onto the target domain
            self._tgt_aligned = self.adata_tgt.obsm[self._tgt_attr]
            self.coupling = self[('src', 'tgt')].solution.transport_matrix
            weights = np.sum(self.coupling, axis = 1)
            self._src_aligned = np.matmul(self.coupling, self._tgt_aligned) / weights[:, None]
        else:
            # Projecting the target domain onto the source domain
            self._src_aligned = self.adata_src.obsm[self._src_attr]
            self.coupling = self[('src', 'tgt')].solution.transport_matrix
            weights = np.sum(self.coupling, axis = 1)
            self._tgt_aligned = np.matmul(np.transpose(self.coupling), self._src_aligned) / weights[:, None]
        
        self.adata_src.obsm["X_aligned"] = self._src_aligned
        self.adata_tgt.obsm["X_aligned"] = self._tgt_aligned
        return self._src_aligned, self._tgt_aligned
    
    def integrate(
            self: CrossModalityIntegrationMixinProtocol[K, B],
            normalize = True,
            norm = "l2",
            SRContoTGT=True,
            **kwargs:Any,
    ) -> ArrayLike:
        """
        Integrate source and target objects.

        Parameters
        ----------
        %(normalize)s
        %(norm)s
        %(SRContoTGT)s

        Returns
        -------
        
        """
        if normalize:
            self._normalize(norm=norm) # überschreibt so die adata objecte, evlt. lieber neues feld in obsm hinzufügen?
        
        src_aligned, tgt_aligned = self._barycentric_projection(SRContoTGT=SRContoTGT)

        self.src_aligned, self.tgt_aligned = src_aligned, tgt_aligned
        return (self.src_aligned, self.tgt_aligned)
                        
    def foscttm(
            self: CrossModalityIntegrationMixinProtocol[K, B],
            **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fraction of samples closer than true match (smaller is better)
        Parameters
        ----------
        **kwargs
            Additional keyword arguments are passed to
            :func:`scipy.spatial.distance_matrix`
        Returns
        -------
        fracs: FOSCTTM for samples in source and target modality
        
        Note
        ----
        Samples in source and target modality should be paired and given in the same order
        """
    
        x = self.adata_src.obsm["X_aligned"] 
        y = self.adata_tgt.obsm["X_aligned"] 
        if x.shape != y.shape:
            raise ValueError("Shapes do not match!")
        d = scipy.spatial.distance_matrix(x, y, **kwargs)
        foscttm_x = (d < np.expand_dims(np.diag(d), axis=1)).mean(axis=1)
        foscttm_y = (d < np.expand_dims(np.diag(d), axis=0)).mean(axis=0)
        fracs = []
        for i in range(len(foscttm_x)):
            fracs.append((foscttm_x[i]+foscttm_y[i])/2)
        return np.mean(fracs).round(4)    

    def plotting(
            self: CrossModalityIntegrationMixinProtocol[K, B],
            color : Union[str, Sequence[str], None] = None, # add cell type here
            **kwargs:Any, 
    ):
        """
        UMAP plot of integrated source and target objects.

        Parameters
        ----------
        %(color)s

        Returns
        -------
        
        """
        adata_comb = ad.concat([self.adata_src, self.adata_tgt], join = 'outer', label='batch', index_unique = '-')
        sc.pp.neighbors(adata_comb, use_rep="X_aligned")
        sc.tl.umap(adata_comb)
        if isinstance(color, str):
            col = ["batch", color]
        elif isinstance(color, list):
            col = ['batch']+ color
        else:
            raise ValueError("Input color must be a string or a list of strings.")

        sc.pl.umap(adata_comb, color=col)
        self.adata_comb = adata_comb

    @property
    def solution(
        self: CrossModalityIntegrationMixinProtocol[K, B],
    ) -> Optional[BaseSolverOutput]:
        """Solution of the optimal transport problem."""
        return self._solution