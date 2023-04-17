from typing import Any, List, Mapping, Optional

import networkx as nx
import numpy as np

from moscot._logging import logger
from moscot._types import ArrayLike
from moscot.base.cost import BaseCost
from moscot.costs._utils import register_cost

__all__ = ["LeafDistance", "BarcodeDistance"]


@register_cost("barcode_distance", backend="moscot")
class BarcodeDistance(BaseCost):
    """Barcode distance."""

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
        logger.info("Computing barcode distance")
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


@register_cost("leaf_distance", backend="moscot")
class LeafDistance(BaseCost):
    """Tree leaf distance."""

    @property
    def data(self) -> nx.Graph:
        try:
            if self._attr != "uns":
                raise NotImplementedError(f"Extracting trees from `adata.{self._attr}` is not yet implemented.")

            tree = self.adata.uns[self._key][self._dist_key]
            if not isinstance(tree, nx.Graph):
                raise TypeError(
                    f"Expected the tree in `adata.uns[{self._key!r}][{self._dist_key!r}]` "
                    f"to be a `networkx.DiGraph`, found `{type(tree)}`."
                )

            return tree
        except KeyError:
            raise KeyError(f"Unable to find tree in `adata.{self._attr}[{self._key!r}][{self._dist_key!r}]`.") from None

    def _compute(
        self,
        **kwargs: Any,
    ) -> ArrayLike:
        logger.info("Computing tree distance")
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
