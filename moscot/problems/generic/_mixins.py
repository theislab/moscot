from typing import Any, Optional, Protocol, TYPE_CHECKING

from typing_extensions import Literal
import pandas as pd

from moscot._types import Filter_t
from moscot.problems.base import AnalysisMixin  # type: ignore[attr-defined]
from moscot._constants._constants import AggregationMode
from moscot.problems.base._mixins import AnalysisMixinProtocol
from moscot.problems.base._compound_problem import B, K


class GenericAnalysisMixinProtocol(AnalysisMixinProtocol[K, B], Protocol[K, B]):
    """Protocol class."""

    batch_key: Optional[str]

    def _cell_transition(  # TODO(@MUCDK) think about removing _cell_transition_non_online
        self: AnalysisMixinProtocol[K, B],
        online: bool,
        *args: Any,
        **kwargs: Any,
    ) -> pd.DataFrame:
        ...


class GenericAnalysisMixin(AnalysisMixin[K, B]):
    """Generic Analysis Mixin."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._batch_key: Optional[str] = None

    def cell_transition(
        self: GenericAnalysisMixinProtocol[K, B],
        source_key: K,
        target_key: K,
        key: Optional[str] = None,
        source_annotation: Filter_t = None,
        target_annotation: Filter_t = None,
        forward: bool = False,  # return value will be row-stochastic if forward=True, else column-stochastic
        aggregation_mode: Literal["annotation", "cell"] = AggregationMode.ANNOTATION,  # type: ignore[assignment]
        online: bool = False,
        other_key: Optional[str] = None,
        other_adata: Optional[str] = None,
        batch_size: Optional[int] = None,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """
        Compute a grouped cell transition matrix.

        This function computes a transition matrix with entries corresponding to categories, e.g. cell types.
        The transition matrix will be row-stochastic if `forward` is `True`, otherwise column-stochastic.

        Parameters
        ----------
        source_key
            Key identifying the source distribution.
        target_key
            Key identifying the target distribution.
        source_annotation
            Can be one of the following:
                - if `source_annotation` is of type :class:`str` this should correspond to a key in
                  :attr:`anndata.AnnData.obs`. In this case, the categories in the transition matrix correspond to the
                  unique values in :attr:`anndata.AnnData.obs` ``['{source_annotation}']``
                - if `target_annotation` is of type :class:`dict`, its key should correspond to a key in
                  :attr:`anndata.AnnData.obs` and its value to a subset of categories present in
                  :attr:`anndata.AnnData.obs` ``['{source_annotation.keys()[0]}']``
        target_annotation
            Can be one of the following
                - if `target_annotation` is of type :class:`str` this should correspond to a key in
                  :attr:`anndata.AnnData.obs`. In this case, the categories in the transition matrix correspond to the
                  unique values in :attr:`anndata.AnnData.obs` ``['{target_annotation}']``
                - if `late_annotation` is of :class:`dict`, its key should correspond to a key in
                  :attr:`anndata.AnnData.obs` and its value to a subset of categories present in
                  :attr:`anndata.AnnData.obs` ``['{target_annotation.keys()[0]}']``
        %(forward_cell_transition)s
        %(aggregation_mode)s
        %(online)s

        Returns
        -------
        Transition matrix of cells or groups of cells.
        """
        if TYPE_CHECKING:
            assert isinstance(self.batch_key, str)
        return self._cell_transition(
            key=self.batch_key,
            source_key=source_key,
            target_key=target_key,
            source_annotation=source_annotation,
            target_annotation=target_annotation,
            forward=forward,
            aggregation_mode=AggregationMode(aggregation_mode),
            online=online,
            other_key=None,
        )

    @property
    def batch_key(self) -> Optional[str]:
        """Return temporal key."""
        return self._batch_key

    @batch_key.setter
    def batch_key(self, value: Optional[str]) -> None:
        if value is not None and value not in self.adata.obs:
            raise KeyError(f"{value} not in `adata.obs`.")
        self._batch_key = value
