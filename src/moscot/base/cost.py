from abc import ABC, abstractmethod
from typing import Any, Tuple, Union

import numpy as np

from anndata import AnnData

from moscot._logging import logger
from moscot._types import ArrayLike

__all__ = ["BaseCost"]


class BaseCost(ABC):
    """Base class for all :mod:`moscot.costs`.

    Parameters
    ----------
    adata
        Annotated data object.
    attr
        Attribute of :class:`~anndata.AnnData` used when computing the cost.
    key
        Key in the ``attr`` of :class:`~anndata.AnnData` used when computing the cost.
    dist_key
        Helper key which determines which distribution :attr:`adata` belongs to.
    """

    def __init__(self, adata: AnnData, attr: str, key: str, dist_key: Union[Any, Tuple[Any, Any]]):
        self._adata = adata
        self._attr = attr
        self._key = key
        self._dist_key = dist_key

    @abstractmethod
    def _compute(self, *args: Any, **kwargs: Any) -> ArrayLike:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> ArrayLike:
        """Compute a cost matrix from :attr:`adata`.

        Parameters
        ----------
        args
            Positional arguments for computation.
        kwargs
            Keyword arguments for computation.

        Returns
        -------
        The computed cost matrix.
        """
        cost = self._compute(*args, **kwargs)
        if np.any(np.isnan(cost)):
            maxx = np.nanmax(cost)
            logger.warning(f"Cost matrix contains `NaN` values, setting them to the maximum value `{maxx}`.")
            cost = np.nan_to_num(cost, nan=maxx)  # type: ignore[call-overload]
        if np.any(cost < 0):
            raise ValueError("Cost matrix contains negative values.")
        return cost

    @property
    def adata(self) -> AnnData:
        """Annotated data object."""
        return self._adata

    # TODO(michalk8): don't require impl.
    @property
    @abstractmethod
    def data(self) -> Any:
        """Container containing the data."""
