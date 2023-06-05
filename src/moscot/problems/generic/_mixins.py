from typing import TYPE_CHECKING, Any, List, Literal, Optional, Protocol, Tuple, Union

import pandas as pd

from anndata import AnnData

from moscot import _constants
from moscot._types import ArrayLike, Str_Dict_t
from moscot.base.problems._mixins import AnalysisMixin, AnalysisMixinProtocol
from moscot.base.problems.compound_problem import ApplyOutput_t, B, K

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
        """Compute a grouped cell transition matrix.

        This function computes a transition matrix with entries corresponding to categories, e.g., cell types.
        The transition matrix will be row-stochastic if ``forward = True``, otherwise column-stochastic.

        Parameters
        ----------
        source
            TODO.
        target
            TODO.
        source_groups
            TODO.
        target_groups
            TODO.
        forward
            TODO.
        aggregation_mode
            TODO.
        batch_size
            TODO.
        normalize
            TODO.
        key_added
            TODO.

        Returns
        -------
        TODO.
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

    def push(
        self: GenericAnalysisMixinProtocol[K, B],
        source: K,
        target: K,
        data: Optional[Union[str, ArrayLike]] = None,
        subset: Optional[Union[str, List[str], Tuple[int, int]]] = None,
        scale_by_marginals: bool = True,
        key_added: Optional[str] = _constants.PUSH,
        return_all: bool = False,
        **kwargs: Any,
    ) -> Optional[ApplyOutput_t[K]]:
        """Push mass from source to target.

        Parameters
        ----------
        source
            Source key in :attr:`solutions`.
        target
            Target key in :attr:`solutions`.
        data
            Initial data to push, see :meth:`~moscot.base.problems.OTProblem.push` for information.
        subset
            Push values contained only within the subset.
        scale_by_marginals
            Whether to scale by the source :term:`marginals`.
        key_added
            Key in :attr:`~anndata.AnnData.obs` where to add the result.
        return_all
            Whether to also return intermediate results. Always true if ``key_added != None``.
        kwargs
            Keyword arguments for the subproblems' :meth:`~moscot.base.problems.OTProblem.push` method.

        Returns
        -------
        Depending on the ``key_added``:

        - :obj:`None` - returns the result.
        - otherwise, returns nothing and updates :attr:`adata.obs['{key_added}'] <anndata.AnnData.obs>`
          with the result.
        """
        # TODO(michalk8): consider not overriding + update the defaults in `BaseCompoundProblem` + implement _post_apply
        data = locals()
        _ = data.pop("kwargs", None)
        return super().push(**data, **kwargs)

    def pull(
        self: GenericAnalysisMixinProtocol[K, B],
        source: K,
        target: K,
        data: Optional[Union[str, ArrayLike]] = None,
        subset: Optional[Union[str, List[str], Tuple[int, int]]] = None,
        scale_by_marginals: bool = True,
        key_added: Optional[str] = _constants.PULL,
        return_all: bool = False,
        **kwargs: Any,
    ) -> Optional[ApplyOutput_t[K]]:
        """Pull mass from target to source.

        Parameters
        ----------
        source
            Source key in :attr:`solutions`.
        target
            Target key in :attr:`solutions`.
        data
            Initial data to pull, see :meth:`~moscot.base.problems.OTProblem.pull` for information.
        subset
            Pull values contained only within the subset.
        scale_by_marginals
            Whether to scale by the source :term:`marginals`.
        key_added
            Key in :attr:`~anndata.AnnData.obs` where to add the result.
        return_all
            Whether to also return intermediate results. Always true if ``key_added != None``.
        kwargs
            Keyword arguments for the subproblems' :meth:`~moscot.base.problems.OTProblem.pull` method.

        Returns
        -------
        Depending on the ``key_added``:

        - :obj:`None` - returns the result.
        - otherwise - returns nothing and updates :attr:`adata.obs['{key_added}'] <anndata.AnnData.obs>`
          with the result.
        """
        # TODO(michalk8): consider not overriding + update the defaults in `BaseCompoundProblem` + implement _post_apply
        data = locals()
        _ = data.pop("kwargs", None)
        return super().pull(**data, **kwargs)

    @property
    def batch_key(self: GenericAnalysisMixinProtocol[K, B]) -> Optional[str]:
        """Batch key in :attr:`~anndata.AnnData.obs`."""
        return self._batch_key

    @batch_key.setter
    def batch_key(self: GenericAnalysisMixinProtocol[K, B], key: Optional[str]) -> None:
        if key is not None and key not in self.adata.obs:
            raise KeyError(f"Unable to find batch data in `adata.obs[{key!r}]`.")
        self._batch_key = key
