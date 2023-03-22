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

class CrossModalityIntegrationMixin(AnalysisMixin[K, B]):
    """Cross modality integration analysis mixin class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._src_attr: Optional[str] = None
        self._tgt_attr: Optional[str] = None
                        
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

    @property
    def solution(self) -> Optional[BaseSolverOutput]:
        """Solution of the optimal transport problem."""
        return self._solution