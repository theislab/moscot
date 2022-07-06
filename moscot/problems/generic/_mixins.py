from typing import Any, Union, Mapping, Optional, Protocol, Sequence, TYPE_CHECKING

from typing_extensions import Literal
import pandas as pd

from moscot.problems.base import AnalysisMixin  # type: ignore[attr-defined]
from moscot._constants._constants import AggregationMode
from moscot.problems.base._mixins import AnalysisMixinProtocol
from moscot.problems.base._compound_problem import B, K


class GenericAnalysisMixinProtocol(AnalysisMixinProtocol[K, B], Protocol[K, B]):
    """Protocol class."""

    batch_key: Optional[str]

    def _cell_transition(
        self: "GenericAnalysisMixinProtocol[K, B]",
        key: str,
        key_source: K,
        key_target: K,
        source_cells: Union[str, Mapping[str, Sequence[Any]]],
        target_cells: Union[str, Mapping[str, Sequence[Any]]],
        forward: bool = False,
        aggregation_mode: Literal["group", "cell"] = AggregationMode.GROUP,  # type: ignore[assignment]
        online: bool = False,
        other_key: Optional[str] = None,
    ) -> pd.DataFrame:
        ...


class GenericAnalysisMixin(AnalysisMixin[K, B]):
    """Generic Analysis Mixin."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._batch_key: Optional[str] = None

    def cell_transition(
        self: GenericAnalysisMixinProtocol[K, B],
        key_source: K,
        key_target: K,
        source_cells: Union[str, Mapping[str, Sequence[Any]]],
        target_cells: Union[str, Mapping[str, Sequence[Any]]],
        forward: bool = False,  # return value will be row-stochastic if forward=True, else column-stochastic
        aggregation_mode: Literal["group", "cell"] = AggregationMode.GROUP,  # type: ignore[assignment]
        online: bool = False,
    ) -> pd.DataFrame:
        """
        Compute a grouped cell transition matrix.

        This function computes a transition matrix with entries corresponding to categories, e.g. cell types.
        The transition matrix will be row-stochastic if `forward` is `True`, otherwise column-stochastic.

        Parameters
        ----------
        key_source
            Key identifying the source distribution.
        key_target
            Key identifying the target distribution.
        source_cells
            Can be one of the following:
                - if `source_cells` is of type :class:`str` this should correspond to a key in
                  :attr:`anndata.AnnData.obs`. In this case, the categories in the transition matrix correspond to the
                  unique values in :attr:`anndata.AnnData.obs` ``['{source_cells}']``
                - if `target_cells` is of type :class:`dict`, its key should correspond to a key in
                  :attr:`anndata.AnnData.obs` and its value to a subset of categories present in
                  :attr:`anndata.AnnData.obs` ``['{source_cells.keys()[0]}']``
        target_cells
            Can be one of the following
                - if `target_cells` is of type :class:`str` this should correspond to a key in
                  :attr:`anndata.AnnData.obs`. In this case, the categories in the transition matrix correspond to the
                  unique values in :attr:`anndata.AnnData.obs` ``['{target_cells}']``
                - if `late_cells` is of :class:`dict`, its key should correspond to a key in
                  :attr:`anndata.AnnData.obs` and its value to a subset of categories present in
                  :attr:`anndata.AnnData.obs` ``['{target_cells.keys()[0]}']``
        forward
            If `True` computes transition from cells belonging to `source_cells` to cells belonging to `target_cells`.
        aggregation_mode:
            If `aggregation_mode` is `group` the transition probabilities from the groups defined by `source_cells` are
            returned. If `aggregation_mode` is `cell` the transition probablities for each cell are returned.
        online
            TODO

        Returns
        -------
        Transition matrix of cells or groups of cells.
        """
        if TYPE_CHECKING:
            assert isinstance(self.batch_key, str)
        return self._cell_transition(
            key=self.batch_key,
            key_source=key_source,
            key_target=key_target,
            source_cells=source_cells,
            target_cells=target_cells,
            forward=forward,
            aggregation_mode=AggregationMode(aggregation_mode),  # type: ignore[arg-type]
            online=online,
            other_key=None,
        )

    @property
    def batch_key(self) -> Optional[str]:
        """Return temporal key."""
        return self._batch_key

    @batch_key.setter
    def batch_key(self, value: Optional[str] = None) -> None:
        self._batch_key = value
