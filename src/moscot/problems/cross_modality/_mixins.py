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


__all__ = ["CrossModalityTranslationMixin"]

class CrossModalityTranslationMixinProtocol(AnalysisMixinProtocol[K, B]):
    """Protocol class."""

    adata_src: AnnData
    adata_tgt: AnnData
    src_attr: Optional[str]
    tgt_attr: Optional[str]

    def _cell_transition(
            self: AnalysisMixinProtocol[K, B], 
            *args: Any, 
            **kwargs: Any
    ) -> pd.DataFrame:
        ...

    def _normalize(
            self: AnalysisMixinProtocol[K, B],
            norm: Literal["l2", "l1", "max"] = "l2", 
            bySample: bool = True
    ) -> None :
        ...

class CrossModalityTranslationMixin(AnalysisMixin[K, B]):
    """Cross modality translation analysis mixin class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._src_attr: Optional[str] = None
        self._tgt_attr: Optional[str] = None

    def translate(
            self: CrossModalityTranslationMixinProtocol[K, B],
            normalize:bool = True,
            norm: Literal["l2", "l1", "max"] = "l2",
            forward:bool = True,
            **kwargs:Any,
    ) -> ArrayLike:
        """
        Translate source or target object.

        Parameters
        ----------
        %(normalize)s
        %(norm)s
        %(forward)s

        Return
        -------
        %(translation)s
        
        """
        if normalize:
            self._normalize(norm=norm) # überschreibt so die adata objecte, evlt. lieber neues feld in obsm hinzufügen?
        
        if forward:
            self._translation = self[('src', 'tgt')].solution.pull(self.adata_tgt.obsm[self._tgt_attr])
        else:
            self._translation = self[('src', 'tgt')].solution.push(self.adata_src.obsm[self._src_attr])

        return self._translation
    
    def _normalize(
            self: CrossModalityTranslationMixinProtocol[K, B],
            norm: Literal["l2", "l1", "max"] = "l2", 
            bySample:bool = True
    )-> None:
        assert (norm in ["l1","l2","max"]), "Norm argument has to be either one of 'max', 'l1', or 'l2'."
        if (bySample==True or bySample==None):
            axis=1
        else:
            axis=0
        self.adata_src.obsm[self._src_attr] = normalize(self.adata_src.obsm[self._src_attr], norm=norm, axis=axis)
        self.adata_tgt.obsm[self._tgt_attr] = normalize(self.adata_tgt.obsm[self._tgt_attr], norm=norm, axis=axis)

    @d_mixins.dedent
    def cell_transition(  # type: ignore[misc]
        self: CrossModalityTranslationMixinProtocol[K, B],
        source: K,
        target: Optional[K] = None,
        source_groups: Optional[Str_Dict_t] = None,
        target_groups: Optional[Str_Dict_t] = None,
        forward: bool = False,  # return value will be row-stochastic if forward=True, else column-stochastic
        aggregation_mode: Literal["annotation", "cell"] = "annotation",
        batch_size: Optional[int] = None,
        normalize: bool = True,
        key_added: Optional[str] = _constants.CELL_TRANSITION,
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
            other_key=None,
            other_adata=self.adata_sc,
            batch_size=batch_size,
            normalize=normalize,
            key_added=key_added,
        )        
