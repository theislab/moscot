from typing import Any, List, Tuple, Union, Literal, Optional, Protocol, TYPE_CHECKING

import pandas as pd
from anndata import AnnData

from moscot._types import ArrayLike, Str_Dict_t
from moscot._docs._docs_mixins import d_mixins
from moscot._constants._constants import Key, AdataKeys, PlottingKeys, PlottingDefaults
from moscot.problems.base._mixins import AnalysisMixin, AnalysisMixinProtocol  # type: ignore[attr-defined]
from moscot.problems.base._compound_problem import B, K, ApplyOutput_t


class GenericAnalysisMixinProtocol(AnalysisMixinProtocol[K, B], Protocol[K, B]):
    """Protocol class."""

    batch_key: Optional[str]
    adata: Optional[AnnData]

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

    @d_mixins.dedent
    def push(
        self: GenericAnalysisMixinProtocol[K, B],
        source: K,
        target: K,
        data: Optional[Union[str, ArrayLike]] = None,
        subset: Optional[Union[str, List[str], Tuple[int, int]]] = None,
        scale_by_marginals: bool = True,
        key_added: Optional[str] = PlottingDefaults.PUSH,
        return_all: bool = False,
        return_data: bool = False,
        **kwargs: Any,
    ) -> Optional[ApplyOutput_t[K]]:
        """
        Push distribution of cells through time.

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
            start=source,
            end=target,
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
            plot_vars = {
                "distribution_key": self.batch_key,
            }
            self.adata.obs[key_added] = self._flatten(result, key=self.batch_key)
            Key.uns.set_plotting_vars(self.adata, AdataKeys.UNS, PlottingKeys.PUSH, key_added, plot_vars)
        if return_data:
            return result

    @d_mixins.dedent
    def pull(
        self: GenericAnalysisMixinProtocol[K, B],
        source: K,
        target: K,
        data: Optional[Union[str, ArrayLike]] = None,
        subset: Optional[Union[str, List[str], Tuple[int, int]]] = None,
        scale_by_marginals: bool = True,
        key_added: Optional[str] = PlottingDefaults.PULL,
        return_all: bool = False,
        return_data: bool = False,
        **kwargs: Any,
    ) -> Optional[ApplyOutput_t[K]]:
        """
        Pull distribution of cells through time.

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
            start=source,
            end=target,
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
            Key.uns.set_plotting_vars(self.adata, AdataKeys.UNS, PlottingKeys.PULL, key_added, plot_vars)
        if return_data:
            return result

    @property
    def batch_key(self) -> Optional[str]:
        """Batch key in :attr:`anndata.AnnData.obs`."""
        return self._batch_key

    @batch_key.setter
    def batch_key(self, key: Optional[str]) -> None:
        if key is not None and key not in self.adata.obs:
            raise KeyError(f"Unable to find batch data in `adata.obs[{key!r}]`.")
        self._batch_key = key
