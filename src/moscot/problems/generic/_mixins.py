from typing import TYPE_CHECKING, Any, List, Literal, Optional, Protocol, Tuple, Union

import pandas as pd

from anndata import AnnData

from moscot import _constants
from moscot._docs._docs_mixins import d_mixins
from moscot._types import ArrayLike, Str_Dict_t
from moscot.base.problems._mixins import AnalysisMixin, AnalysisMixinProtocol
from moscot.base.problems.compound_problem import ApplyOutput_t, B, K
from moscot.plotting._utils import set_plotting_vars

__all__ = ["GenericAnalysisMixin"]


class GenericAnalysisMixinProtocol(AnalysisMixinProtocol[K, B], Protocol[K, B]):
    """Protocol class."""

    _batch_key: Optional[str]
    batch_key: Optional[str]
    adata: AnnData

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

    @d_mixins.dedent
    def push(
        self: GenericAnalysisMixinProtocol[K, B],
        source: K,
        target: K,
        data: Optional[Union[str, ArrayLike]] = None,
        subset: Optional[Union[str, List[str], Tuple[int, int]]] = None,
        scale_by_marginals: bool = True,
        key_added: Optional[str] = _constants.PUSH,
        return_all: bool = False,
        return_data: bool = False,
        **kwargs: Any,
    ) -> Optional[ApplyOutput_t[K]]:
        """
        Push distribution of cells from source to target.

        Parameters
        ----------
        %(start)s
        %(end)s
        %(data)s
        %(subset)s
        %(scale_by_marginals)s
        %(key_added_plotting)s
        %(return_all)s
        %(return_data)s

        Return
        ------
        %(return_push_pull)s

        """
        result = self._apply(
            source=source,
            target=target,
            data=data,
            subset=subset,
            forward=True,
            return_all=return_all or key_added is not None,
            scale_by_marginals=scale_by_marginals,
            **kwargs,
        )

        if TYPE_CHECKING:
            assert isinstance(result, dict)

        if key_added is not None:
            if TYPE_CHECKING:
                assert isinstance(key_added, str)
            plot_vars = {
                "distribution_key": self.batch_key,
            }
            self.adata.obs[key_added] = self._flatten(result, key=self.batch_key)
            set_plotting_vars(self.adata, _constants.PUSH, key=key_added, value=plot_vars)
        return result if return_data else None

    @d_mixins.dedent
    def pull(
        self: GenericAnalysisMixinProtocol[K, B],
        source: K,
        target: K,
        data: Optional[Union[str, ArrayLike]] = None,
        subset: Optional[Union[str, List[str], Tuple[int, int]]] = None,
        scale_by_marginals: bool = True,
        key_added: Optional[str] = _constants.PULL,
        return_all: bool = False,
        return_data: bool = False,
        **kwargs: Any,
    ) -> Optional[ApplyOutput_t[K]]:
        """
        Pull distribution of cells from target to source.

        Parameters
        ----------
        %(source)s
        %(target)s
        %(data)s
        %(subset)s
        %(scale_by_marginals)s
        %(key_added_plotting)s
        %(return_all)s
        %(return_data)s

        Return
        ------
        %(return_push_pull)s

        """
        result = self._apply(
            source=source,
            target=target,
            data=data,
            subset=subset,
            forward=False,
            return_all=return_all or key_added is not None,
            scale_by_marginals=scale_by_marginals,
            **kwargs,
        )
        if TYPE_CHECKING:
            assert isinstance(result, dict)

        if key_added is not None:
            plot_vars = {
                "key": self.batch_key,
            }
            self.adata.obs[key_added] = self._flatten(result, key=self.batch_key)
            set_plotting_vars(self.adata, _constants.PULL, key=key_added, value=plot_vars)
        return result if return_data else None

    @property
    def batch_key(self: GenericAnalysisMixinProtocol[K, B]) -> Optional[str]:
        """Batch key in :attr:`anndata.AnnData.obs`."""
        return self._batch_key

    @batch_key.setter
    def batch_key(self: GenericAnalysisMixinProtocol[K, B], key: Optional[str]) -> None:
        if key is not None and key not in self.adata.obs:
            raise KeyError(f"Unable to find batch data in `adata.obs[{key!r}]`.")
        self._batch_key = key
