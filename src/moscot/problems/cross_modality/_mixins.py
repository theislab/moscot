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

class CrossModalityTranslationMixin(AnalysisMixin[K, B]):
    """Cross modality translation analysis mixin class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._src_attr: Optional[str] = None
        self._tgt_attr: Optional[str] = None

    def translate(
            self: CrossModalityTranslationMixinProtocol[K, B],
            source: K,
            target: K,
            forward: bool = True,
            **kwargs: Any,
    ) -> ArrayLike:
        """
        Translate source or target object.

        Parameters
        ----------
        source
            Key identifying the source distribution.
        target
            Key identifying the target distribution.
        forward
            If `True` computes translation from source object to target object, otherwise backward.
        kwargs
            keyword arguments for policy-specific `_apply` method of :class:`moscot.problems.CompoundProblem`.
        
        """
        if forward:
            self._translation = self[(source, target)].pull(self.adata_tgt.obsm[self._tgt_attr], scale_by_marginals=True, normalize=False, **kwargs)
        else:
            if self.batch_key is None:
                self._translation = self[(source, target)].push(self.adata_src.obsm[self._src_attr], scale_by_marginals=True, normalize=False, **kwargs)
            else:
                self._translation = self[(source, target)].push(self[(source, target)].adata_src.obsm[self._src_attr], scale_by_marginals=True, normalize=False, **kwargs)
        
        return self._translation

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
        source
            Key identifying the source distribution.
        target
            Key identifying the target distribution.
        source_groups
            Can be one of the following:

                - if `source_groups` is of type :class:`str` this should correspond to a key in :attr:`anndata.AnnData.obs`. 
                  In this case, the categories in the transition matrix correspond to the unique values in 
                  :attr:`anndata.AnnData.obs` ``['{source_groups}']``.
                - if `target_groups` is of type :class:`dict`, its key should correspond to a key in :attr:`anndata.AnnData.obs` 
                  and its value to a list containing a subset of categories present in :attr:`anndata.AnnData.obs` ``['{source_groups.keys()[0]}']``. 
                  The order of the list determines the order in the transition matrix.
        target_groups
            Can be one of the following:

                - if `target_groups` is of type :class:`str` this should correspond to a key in
                  :attr:`anndata.AnnData.obs`. In this case, the categories in the transition matrix correspond to the
                  unique values in :attr:`anndata.AnnData.obs` ``['{target_groups}']``.
                - if `target_groups` is of :class:`dict`, its key should correspond to a key in
                  :attr:`anndata.AnnData.obs` and its value to a list containing a subset of categories present in
                  :attr:`anndata.AnnData.obs` ``['{target_groups.keys()[0]}']``. The order of the list determines the order
                  in the transition matrix.
        forward
            If `True` computes transition from `source_annotations` to `target_annotations`, otherwise backward.
        aggregation_mode
            - `annotation`: transition probabilities from the groups defined by `source_annotation` are returned.
            - `cell`: the transition probabilities for each cell are returned.
        batch_size
            number of data points the matrix-vector products are applied to at the same time. The larger, the more memory is required.
        normalize
            If `True` the transition matrix is normalized such that it is stochastic. If `forward` is `True`, the transition 
            matrix is row-stochastic, otherwise column-stochastic.
        key_added
            Key in :attr:`anndata.AnnData.uns` and/or :attr:`anndata.AnnData.obs` where the results
            for the corresponding plotting functions are stored.

        Returns
        -------
        Transition matrix of cells or groups of cells.

        Notes
        -----
        To visualise the results, see :func:`moscot.pl.cell_transition`.
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
            other_adata=self.adata_tgt,
            batch_size=batch_size,
            normalize=normalize,
            key_added=key_added,
        )        
