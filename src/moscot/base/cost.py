import abc
from typing import Any, Optional, Tuple, Union

import numpy as np

from anndata import AnnData

from moscot._logging import logger
from moscot._types import ArrayLike

__all__ = ["BaseCost"]


class BaseCost(abc.ABC):
    """Base class for :mod:`moscot.costs`.

    Parameters
    ----------
    adata
        Annotated data object.
    attr
        Attribute of :class:`~anndata.AnnData`.
    key
        Key in the attribute of :class:`~anndata.AnnData`.
    dist_key
        Key which determines into which source/target subset ``adata`` belongs.
        Useful when :attr:`attr = 'uns' <anndata.AnnData.uns>`.
    """

    def __init__(self, adata: AnnData, attr: str, key: str, dist_key: Optional[Union[Any, Tuple[Any, Any]]] = None):
        self._adata = adata
        self._attr = attr
        self._key = key
        self._dist_key = dist_key

    @abc.abstractmethod
    def _compute(self, *args: Any, **kwargs: Any) -> ArrayLike:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> ArrayLike:
        """Compute the cost matrix.

        Parameters
        ----------
        args
            Positional arguments.
        kwargs
            Keyword arguments.

        Returns
        -------
        The cost matrix.
        """
        cost = self._compute(*args, **kwargs)
        if np.any(np.isnan(cost)):
            maxx = np.nanmax(cost)
            logger.warning(
                f"Cost matrix contains `{np.sum(np.isnan(cost))}` NaN values, "
                f"setting them to the maximum value `{maxx}`."
            )
            cost = np.nan_to_num(cost, nan=maxx)  # type: ignore[call-overload]
        if np.any(cost < 0):
            raise ValueError(f"Cost matrix contains `{np.sum(cost < 0)}` negative values.")
        return cost

    @property
    def adata(self) -> AnnData:
        """Annotated data object."""
        return self._adata
