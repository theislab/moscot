from typing import Any, Literal, Optional, Protocol, TYPE_CHECKING

import pandas as pd

from moscot._types import Str_Dict_t
from moscot.problems.base import AnalysisMixin  # type: ignore[attr-defined]
from moscot._docs._docs_mixins import d_mixins
from moscot._constants._constants import PlottingDefaults
from moscot.problems.base._mixins import AnalysisMixinProtocol
from moscot.problems.base._compound_problem import B, K


class GenericAnalysisMixinProtocol(AnalysisMixinProtocol[K, B], Protocol[K, B]):
    """Protocol class."""

    batch_key: Optional[str]

    def _cell_transition(
        self: AnalysisMixinProtocol[K, B],
        *args: Any,
        **kwargs: Any,
    ) -> pd.DataFrame:
        ...


class GenericAnalysisMixin(AnalysisMixin[K, B]):
    """Generic Analysis Mixin."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._batch_key: Optional[str] = None

    @d_mixins.dedent
    def cell_transition(
        self: GenericAnalysisMixinProtocol[K, B],
        source: K,
        target: K,
        source_groups: Optional[Str_Dict_t] = None,
        target_groups: Optional[Str_Dict_t] = None,
        key: Optional[str] = None,
        forward: bool = False,  # return value will be row-stochastic if forward=True, else column-stochastic
        aggregation_mode: Literal["annotation", "cell"] = "annotation",
        batch_size: Optional[int] = None,
        normalize: bool = True,
        key_added: Optional[str] = PlottingDefaults.CELL_TRANSITION,
    ) -> pd.DataFrame:
        """
        Compute a grouped cell transition matrix.

        This function computes a transition matrix with entries corresponding to categories, e.g. cell types.
        The transition matrix will be row-stochastic if `forward` is `True`, otherwise column-stochastic.

        Parameters
        ----------
        %(cell_trans_params)s
        %(key)s
        %(forward_cell_transition)s
        %(aggregation_mode)s
        %(other_key)s
        %(other_adata)s
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
            assert isinstance(self.batch_key, str)
        return self._cell_transition(
            key=self.batch_key,
            source=source,
            target=target,
            source_groups=source_groups,
            target_groups=target_groups,
            forward=forward,
            aggregation_mode=aggregation_mode,
            batch_size=batch_size,
            normalize=normalize,
            other_key=None,
            key_added=key_added,
        )

    @property
    def batch_key(self) -> Optional[str]:
        """Batch key in :attr:`anndata.AnnData.obs`."""
        return self._batch_key

    @batch_key.setter
    def batch_key(self, key: Optional[str]) -> None:
        if key is not None and key not in self.adata.obs:
            raise KeyError(f"Unable to find batch data in `adata.obs[{key!r}]`.")
        self._batch_key = key
