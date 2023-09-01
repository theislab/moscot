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
        self._batch_key: Optional[str] = None

    def translate(  # type: ignore[misc]
        self: CrossModalityTranslationMixinProtocol[K, B],
        source: K,
        target: K,
        forward: bool = True,
        alternative_attr: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ArrayLike:
        """Translate the source modality to the target modality.

        .. seealso::
            - See :doc:`../notebooks/tutorials/600_tutorial_translation` on how to translate
              `chromatic accessibility <https://en.wikipedia.org/wiki/ATAC-seq>`_ to
              `gene expression <https://en.wikipedia.org/wiki/Single-cell_sequencing>`_.

        Parameters
        ----------
        source
            Key identifying the source distribution.
        target
            Key identifying the target distribution.
        forward
            If :obj:`True`, translate the ``source`` modality to the ``target`` modality, otherwise vice-versa.
        alternative_attr
            Alternative embedding to translate. Valid option are:

            - :obj:`None` - use the features specified when :meth:`preparing <prepare>` the problem.
            - :class:`str` - key in :attr:`~anndata.AnnData.obsm` where the data is stored.
            - :class:`dict`-  it should contain ``'attr'`` and ``'key'``, the attribute and the key
              in :class:`~anndata.AnnData`.
        kwargs
            Keyword arguments for :meth:`push` or :meth:`pull`, depending on the ``forward``.

        Returns
        -------
        If ``forward = True``, the translation of the ``source`` modality to the ``target`` modality,
        otherwise vice-versa.
        """

        def _get_features(
            adata: AnnData,
            attr: Dict[str, Any],
        ) -> ArrayLike:
            data = getattr(adata, attr["attr"])
            key = attr.get("key", None)
            return data if key is None else data[key]

        if self._src_attr is None:
            raise ValueError("Source attribute is `None`.")
        if self._tgt_attr is None:
            raise ValueError("Target attribute is `None`.")

        kwargs["scale_by_marginals"] = True
        kwargs["normalize"] = False

        prob = self[source, target]  # type: ignore[index]
        if alternative_attr is None:
            src_attr = self._src_attr
            tgt_attr = self._tgt_attr
        elif isinstance(alternative_attr, str):
            src_attr = tgt_attr = {"attr": "obsm", "key": alternative_attr}
        else:
            src_attr = tgt_attr = alternative_attr

        if forward:
            return prob.pull(_get_features(self.adata_tgt, attr=tgt_attr), **kwargs)

        adata_src = self.adata_src if self.batch_key is None else prob.adata_src
        return prob.push(_get_features(adata_src, attr=src_attr), **kwargs)

    def cell_transition(  # type: ignore[misc]
        self: CrossModalityTranslationMixinProtocol[K, B],
        source: K,
        target: Optional[K] = None,
        source_groups: Optional[Str_Dict_t] = None,
        target_groups: Optional[Str_Dict_t] = None,
        aggregation_mode: Literal["annotation", "cell"] = "annotation",
        forward: bool = False,
        batch_size: Optional[int] = None,
        normalize: bool = True,
        key_added: Optional[str] = _constants.CELL_TRANSITION,
    ) -> pd.DataFrame:
        """Aggregate the transport matrix.

        This function computes a transition matrix with entries corresponding to categories, e.g., cell types.

        .. seealso::
            - See :doc:`../notebooks/examples/plotting/200_cell_transitions`
              on how to compute and plot the cell transitions.

        Parameters
        ----------
        source
            Key identifying the source distribution.
        target
            Key identifying the target distribution. If :obj:`None`, use the reference.
        source_groups
            Source groups used for aggregation. Valid options are:

            - :class:`str` - key in :attr:`~anndata.AnnData.obs` where categorical data is stored.
            - :class:`dict` - a dictionary with one key corresponding to a categorical column in
              :attr:`~anndata.AnnData.obs` and values to a subset of categories.
        target_groups
            Target groups used for aggregation. Valid options are:

            - :class:`str` - key in :attr:`~anndata.AnnData.obs` where categorical data is stored.
            - :class:`dict` - a dictionary with one key corresponding to a categorical column in
              :attr:`~anndata.AnnData.obs` and values to a subset of categories.
        aggregation_mode
            How to aggregate the cell-level transport maps. Valid options are:

            - ``'annotation'`` - group the transitions by the ``source_groups`` and the ``target_groups``.
            - ``'cell'`` - TODO.
        forward
            If :obj:`True`, compute the transitions from the ``source_groups`` to the ``target_groups``.
        batch_size
            Number of rows/columns of the cost matrix to materialize during :meth:`push` or :meth:`pull`.
            Larger value will require more memory.
        normalize
            If :obj:`True`, normalize the transition matrix. If ``forward = True``, the transition matrix
            will be row-stochastic, otherwise column-stochastic.
        key_added
            Key in :attr:`~anndata.AnnData.uns` where to save the result.

        Returns
        -------
        Depending on the ``key_added``:

        - :obj:`None` - returns the transition matrix.
        - :obj:`str` - returns nothing and saves the transition matrix to
          :attr:`uns['moscot_results']['cell_transition']['{key_added}'] <anndata.AnnData.uns>`
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

    @property
    def batch_key(self) -> Optional[str]:
        """Batch key in :attr:`~anndata.AnnData.obs`."""
        return self._batch_key

    @batch_key.setter
    def batch_key(self, key: Optional[str]) -> None:
        if key is not None and key not in self.adata.obs:  # type: ignore[attr-defined]
            raise KeyError(f"Unable to find batch data in `adata.obs[{key!r}]`.")
        self._batch_key = key
