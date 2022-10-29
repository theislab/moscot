from abc import ABC, abstractmethod
from typing import Any, List, Literal, Mapping, Optional

import networkx as nx

import numpy as np

from anndata import AnnData

from moscot._types import ArrayLike
from moscot._logging import logger
from moscot._docs._docs import d

__all__ = ["BaseCost", "LeafDistance", "BarcodeDistance"]


@d.dedent
class BaseCost(ABC):
    """Base class for all :mod:`moscot` losses.

    Parameters
    ----------
    %(adata)s
    attr
        Attribute of :attr:`adata` to access when computing the cost.
    key
        Key in in the ``attr`` of :attr:`adata` to access when computing the cost.
    """

    def __init__(self, adata: AnnData, attr: str, key: Optional[str] = None):
        self._adata = adata
        self._attr = attr
        self._key = key

    @abstractmethod
    def _compute(self, *args: Any, **kwargs: Any) -> ArrayLike:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> ArrayLike:
        """Compute a cost matrix from :attr:`adata`.

        Parameters
        ----------
        args
            Positional arguments for :meth:`_compute`.
        kwargs
            Keyword arguments for :meth:`_compute`.

        Returns
        -------
        The computed cost matrix.
        """
        cost = self._compute(*args, **kwargs)
        if not np.all(np.isnan(cost)):
            maxx = np.nanmax(cost)
            logger.warning(f"Cost matrix contains `NaN` values, setting them to the maximum value `{maxx}`.")
            cost = np.nan_to_num(cost, nan=maxx)  # type: ignore[call-overload]
        return cost

    @classmethod
    def create(cls, kind: Literal["leaf_distance", "barcode_distance"], *args: Any, **kwargs: Any) -> "BaseCost":
        """Create :mod:`moscot` cost instance.

        Parameters
        ----------
        kind
            Kind of the cost matrix to create.
        args
            Positional arguments for :class:`moscot.costs.BaseCost`.
        kwargs
            Keyword arguments for :class:`moscot.costs.BaseCost`.

        Returns
        -------
        The base cost instance.
        """
        if kind == "leaf_distance":
            return LeafDistance(*args, **kwargs)
        if kind == "barcode_distance":
            return BarcodeDistance(*args, **kwargs)
        raise NotImplementedError(f"Cost function `{kind}` is not yet implemented.")

    @property
    def adata(self) -> AnnData:
        """Annotated data object."""
        return self._adata

    @property
    @abstractmethod
    def data(self) -> Any:
        """Container containing the data."""


class BarcodeDistance(BaseCost):
    """Barcode distances."""

    @property
    def data(self) -> ArrayLike:
        try:
            container = getattr(self.adata, self._attr)
            return container[self._key]
        except AttributeError:
            raise AttributeError(f"`Anndata` has no attribute `{self._attr}`.") from None
        except KeyError:
            raise KeyError(f"Unable to find data in `adata.{self._attr}[{self._key!r}]`.") from None

    def _compute(
        self,
        *_: Any,
        **__: Any,
    ) -> ArrayLike:
        barcodes = self.data
        n_cells = barcodes.shape[0]
        distances = np.zeros((n_cells, n_cells))
        for i in range(n_cells):
            distances[i, i + 1 :] = [
                self._scaled_hamming_dist(barcodes[i, :], barcodes[j, :]) for j in range(i + 1, n_cells)
            ]
        return distances + np.transpose(distances)

    @staticmethod
    def _scaled_hamming_dist(x: ArrayLike, y: ArrayLike) -> float:
        """Adapted from `LineageOT <https://github.com/aforr/LineageOT/>`_."""
        shared_indices = (x >= 0) & (y >= 0)
        b1 = x[shared_indices]

        # there may not be any sites where both were measured
        if not len(b1):
            return np.nan
        b2 = y[shared_indices]

        differences = b1 != b2
        double_scars = differences & (b1 != 0) & (b2 != 0)

        return (np.sum(differences) + np.sum(double_scars)) / len(b1)


class LeafDistance(BaseCost):
    """Tree leaf distances."""

    @property
    def data(self) -> nx.Graph:
        try:
            if self._attr != "uns":
                raise NotImplementedError(f"Extracting trees from `adata.{self._attr}` is not yet implemented.")

            tree = self.adata.uns["trees"][self._key]
            if not isinstance(tree, nx.Graph):
                raise TypeError(
                    f"Expected the tree in `adata.uns['trees'][{self._key!r}]` "
                    f"to be a `networkx.DiGraph`, found `{type(tree)}`."
                )
            return tree
        except KeyError:
            raise KeyError(f"Unable to find tree in `adata.{self._attr}['trees'][{self._key!r}]`.") from None

    def _compute(
        self,
        **kwargs: Any,
    ) -> ArrayLike:
        tree = self.data
        undirected_tree = tree.to_undirected()
        leaves = self._get_leaves(undirected_tree)
        n_leaves = len(leaves)

        distances = np.zeros((n_leaves, n_leaves), dtype=np.float_)
        for i, leaf in enumerate(leaves):
            # TODO(@MUCDK): more efficient, problem: `target`in `multi_source_dijkstra` cannot be chosen as a subset
            distance_dictionary = nx.multi_source_dijkstra(undirected_tree, [leaf], **kwargs)[0]
            distances[i, :] = [distance_dictionary.get(leaf) for leaf in leaves]

        return distances

    def _get_leaves(self, tree: nx.Graph, cell_to_leaf: Optional[Mapping[str, Any]] = None) -> List[Any]:
        leaves = [node for node in tree if tree.degree(node) == 1]
        if not set(self.adata.obs_names).issubset(leaves):
            if cell_to_leaf is None:
                raise ValueError("Leaves do not match `AnnData`'s observation names, please specify `cell_to_leaf`.")
            return [cell_to_leaf[cell] for cell in self.adata.obs.index]
        return [cell for cell in self.adata.obs_names if cell in leaves]
