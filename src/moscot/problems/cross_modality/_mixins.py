from typing import TYPE_CHECKING, Any, Dict, Literal, Optional

import pandas as pd

from anndata import AnnData

from moscot import _constants
from moscot._types import ArrayLike, Str_Dict_t
from moscot.base.problems._mixins import AnalysisMixin, AnalysisMixinProtocol
from moscot.base.problems.compound_problem import B, K

__all__ = ["CrossModalityTranslationMixin"]


class CrossModalityTranslationMixinProtocol(AnalysisMixinProtocol[K, B]):
    """Protocol class."""

    adata_src: AnnData
    adata_tgt: AnnData
    _src_attr: Optional[Dict[str, Any]]
    _tgt_attr: Optional[Dict[str, Any]]
    batch_key: Optional[str]

    def _cell_transition(self: AnalysisMixinProtocol[K, B], *args: Any, **kwargs: Any) -> pd.DataFrame:
        ...


class CrossModalityTranslationMixin(AnalysisMixin[K, B]):
    """Cross modality translation analysis mixin class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._src_attr: Optional[Dict[str, Any]] = None
        self._tgt_attr: Optional[Dict[str, Any]] = None

    def translate(  # type: ignore[misc]
        self: CrossModalityTranslationMixinProtocol[K, B],
        source: K,
        target: K,
        forward: bool = True,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> ArrayLike:
        """Translate source modality to target modality.

        Parameters
        ----------
        source
            Key identifying the source distribution.
        target
            Key identifying the target distribution.
        forward
            If `True`, compute the translation from :attr:`adata_src` to :attr:`adata_tgt`, otherwise vice-versa.
        kwargs
            Keyword arguments for policy-specific `_apply` method of :class:`~moscot.base.problems.CompoundProblem`.

        Returns
        -------
        Translation from :attr:`adata_src` in target domain or from :attr:`adata_tgt` in source domain,
        depending on `forward`.
        """
        if self._src_attr is None:
            raise ValueError("source attribute is None")
        if self._tgt_attr is None:
            raise ValueError("target attribute is None")

        def _get_features(
            adata: AnnData,
            attr: Optional[Dict[str, Any]] = None,
        ) -> ArrayLike:
            attr = {"attr": "X"} if attr is None else attr
            att = attr.get("attr", None)
            key = attr.get("key", None)

            if key is not None:
                return getattr(adata, att)[key]
            return getattr(adata, att)

        kwargs["scale_by_marginals"] = True
        kwargs["normalize"] = False

        if forward:
            return self[source, target].pull(  # type: ignore[index]
                _get_features(adata=self.adata_tgt, attr=self._tgt_attr),
                **kwargs,
            )
        if self.batch_key is None:
            return self[source, target].push(  # type: ignore[index]
                _get_features(adata=self.adata_src, attr=self._src_attr), **kwargs
            )
        return self[source, target].push(  # type: ignore[index]
            _get_features(adata=self[source, target].adata_src, attr=self._src_attr),  # type: ignore[index]
            **kwargs,
        )

    def cell_transition(  # type: ignore[misc]
        self: CrossModalityTranslationMixinProtocol[K, B],
        source: K,
        target: Optional[K] = None,
        source_groups: Optional[Str_Dict_t] = None,
        target_groups: Optional[Str_Dict_t] = None,
        forward: bool = False,
        aggregation_mode: Literal["annotation", "cell"] = "annotation",
        batch_size: Optional[int] = None,
        normalize: bool = True,
        key_added: Optional[str] = _constants.CELL_TRANSITION,
    ) -> pd.DataFrame:
        """Compute a grouped cell transition matrix.

        This function computes a transition matrix with entries corresponding to categories, e.g. cell types.
        The transition matrix will be row-stochastic if `forward` is `True`, otherwise column-stochastic.

        Parameters
        ----------
        source
            Key defining the batches in the source distribution.
        target
            Key defining the batches in the target distribution.
        source_groups
            Can be one of the following:

                - if ``source_groups`` is of type :class:`str` this should correspond to a key in
                  :attr:`anndata.AnnData.obs`.
                  In this case, the categories in the transition matrix correspond to the unique values in
                  :attr:`anndata.AnnData.obs` ``['{source_groups}']``.
                - if ``target_groups`` is of type :class:`dict`, its key should correspond to a key in
                  :attr:`anndata.AnnData.obs` and its value to a list containing a subset of categories
                  present in :attr:`anndata.AnnData.obs` ``['{source_groups.keys()[0]}']``.
                  The order of the list determines the order in the transition matrix.
        target_groups
            Can be one of the following:

                - if ``target_groups`` is of type :class:`str` this should correspond to a key in
                  :attr:`anndata.AnnData.obs`. In this case, the categories in the transition matrix correspond to the
                  unique values in :attr:`anndata.AnnData.obs` ``['{target_groups}']``.
                - if ``target_groups`` is of :class:`dict`, its key should correspond to a key in
                  :attr:`anndata.AnnData.obs` and its value to a list containing a subset of categories present in
                  :attr:`anndata.AnnData.obs` ``['{target_groups.keys()[0]}']``. The order of the list determines the
                  order in the transition matrix.
        forward
            If `True` computes transition from `source_annotations` to `target_annotations`, otherwise backward.
        aggregation_mode
            - `annotation`: transition probabilities from the groups defined by `source_annotation` are returned.
            - `cell`: the transition probabilities for each cell are returned.
        batch_size
            number of data points the matrix-vector products are applied to at the same time. The larger,
            the more memory is required.
        normalize
            If `True` the transition matrix is normalized such that it is stochastic. If `forward` is `True`, the
            transition matrix is row-stochastic, otherwise column-stochastic.
        key_added
            Key in :attr:`~anndata.AnnData.uns` and/or :attr:`~anndata.AnnData.obs` where the results
            for the corresponding plotting functions are stored.

        Returns
        -------
        Transition matrix of cells or groups of cells.

        Notes
        -----
        To visualise the results, see :func:`~moscot.pl.cell_transition`.
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
