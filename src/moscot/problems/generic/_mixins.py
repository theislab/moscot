from typing import TYPE_CHECKING, Any, List, Literal, Optional, Protocol, Tuple, Union

import pandas as pd

from anndata import AnnData

from moscot import _constants
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

    def cell_transition(
        self: GenericAnalysisMixinProtocol[K, B],
        source: K,
        target: K,
        source_groups: Optional[Str_Dict_t] = None,
        target_groups: Optional[Str_Dict_t] = None,
        aggregation_mode: Literal["annotation", "cell"] = "annotation",
        forward: bool = False,
        batch_size: Optional[int] = None,
        normalize: bool = True,
        key_added: Optional[str] = _constants.CELL_TRANSITION,
    ) -> pd.DataFrame:
        """Aggregate the transport matrix.

        .. seealso::
            - See :doc:`../notebooks/examples/plotting/200_cell_transitions`
              on how to compute and plot the cell transitions.

        Parameters
        ----------
        source
            Key identifying the source distribution.
        target
            Key identifying the target distribution.
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
            - ``'cell'`` - do not group by the ``source_groups`` or the ``target_groups``, depending on the ``forward``.
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
            assert isinstance(self.batch_key, str)
        # TODO(michalk8): modify the behavior to match with the docs
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
        - :class:`str` - returns nothing and updates :attr:`obs['{key_added}'] <anndata.AnnData.obs>`
          with the result.
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
            return None
        return result

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
        - :class:`str` - returns nothing and updates :attr:`obs['{key_added}'] <anndata.AnnData.obs>`
          with the result.
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
            return None
        return result

    @property
    def batch_key(self: GenericAnalysisMixinProtocol[K, B]) -> Optional[str]:
        """Batch key in :attr:`~anndata.AnnData.obs`."""
        return self._batch_key

    @batch_key.setter
    def batch_key(self: GenericAnalysisMixinProtocol[K, B], key: Optional[str]) -> None:
        if key is not None and key not in self.adata.obs:
            raise KeyError(f"Unable to find batch data in `adata.obs[{key!r}]`.")
        self._batch_key = key
