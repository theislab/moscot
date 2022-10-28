from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union, Literal, Mapping, Optional

import networkx as nx

import numpy as np

from anndata import AnnData

from moscot._types import ArrayLike

__all__ = ["LeafDistance", "BarcodeDistance"]
Scale_t = Literal["max", "min", "median", "mean"]


class BaseLoss(ABC):
    """Base class handling all :mod:`moscot` losses."""

    @abstractmethod
    def _compute(self, *args: Any, **kwargs: Any) -> ArrayLike:
        pass

    def __init__(self, adata: AnnData, attr: str, key: str, dist_key: Union[Any, Tuple[Any, Any]]):
        self._adata = adata
        self._attr = attr
        self._key = key
        self._dist_key = dist_key

    def __call__(self, *args: Any, **kwargs: Any) -> ArrayLike:
        return self._compute(*args, **kwargs)

    @classmethod
    def create(cls, kind: Literal["leaf_distance", "barcode_distance"], *args: Any, **kwargs: Any) -> "BaseLoss":
        if kind == "leaf_distance":
            return LeafDistance(*args, **kwargs)
        if kind == "barcode_distance":
            return BarcodeDistance(*args, **kwargs)
        raise NotImplementedError(f"Cost function `{kind}` is not yet implemented.")


class BarcodeDistance(BaseLoss):
    """Class handling Barcode distances."""

    def _compute(
        self,
        *_: Any,
        **__: Any,
    ) -> ArrayLike:
        container = getattr(self._adata, self._attr)
        if self._key not in container:
            raise KeyError(f"Unable to find data in `adata.{self._attr}[{self._key!r}]`.")
        barcodes = container[self._key]
        n_cells = barcodes.shape[0]
        distances = np.zeros((n_cells, n_cells))
        for i in range(n_cells):
            distances[i, i + 1 :] = [
                self._scaled_Hamming_distance(barcodes[i, :], barcodes[j, :]) for j in range(i + 1, n_cells)
            ]
        return distances + np.transpose(distances)

    @staticmethod
    def _scaled_Hamming_distance(x: ArrayLike, y: ArrayLike) -> float:
        """
        adapted from https://github.com/aforr/LineageOT/blob/8c66c630d61da289daa80e29061e888b1331a05a/lineageot/inference.py#L33.  # noqa: E501
        """

        shared_indices = (x >= 0) & (y >= 0)
        b1 = x[shared_indices]

        # There may not be any sites where both were measured
        if not len(b1):
            # TODO(@MUCDK): What to do if this happens? set to maximum or raise, depending on what user wants
            return np.nan
        b2 = y[shared_indices]

        differences = b1 != b2
        double_scars = differences & (b1 != 0) & (b2 != 0)

        return (np.sum(differences) + np.sum(double_scars)) / len(b1)


class LeafDistance(BaseLoss):
    """Class handling leaf distances (from trees)."""

    def _compute(
        self,
        **kwargs: Any,
    ) -> ArrayLike:
        """
        Compute the matrix of pairwise distances between leaves of the tree
        """
        if self._attr == "uns":
            tree = self._adata.uns[self._key][self._dist_key]
            if not isinstance(tree, nx.DiGraph):
                raise TypeError(
                    f"Expected the tree in `adata.uns['trees'][{self._dist_key!r}]` "
                    f"to be a `networkx.DiGraph`, found `{type(tree)}`."
                )
            return self._create_cost_from_tree(tree, **kwargs)
        raise NotImplementedError(f"Extracting trees from `adata.{self._attr}` is not implemented.")

    def _create_cost_from_tree(self, tree: nx.DiGraph, **kwargs: Any) -> ArrayLike:
        # TODO(@MUCDK): more efficient, problem: `target`in `multi_source_dijkstra` cannot be chosen as a subset
        undirected_tree = tree.to_undirected()
        leaves = self._get_leaves(undirected_tree)
        n_leaves = len(leaves)
        distances = np.zeros((n_leaves, n_leaves))
        for i, leaf in enumerate(leaves):
            distance_dictionary = nx.multi_source_dijkstra(undirected_tree, [leaf], **kwargs)[0]
            distances[i, :] = [distance_dictionary.get(leaf) for leaf in leaves]
        return distances

    def _get_leaves(self, tree: nx.DiGraph, cell_to_leaf: Optional[Mapping[str, Any]] = None) -> List[Any]:
        leaves = [node for node in tree if tree.degree(node) == 1]
        if not set(self._adata.obs_names).issubset(leaves):
            if cell_to_leaf is None:
                raise ValueError("Leaves do not match `AnnData`'s observation names, please specify `cell_to_leaf`.")
            return [cell_to_leaf[cell] for cell in self._adata.obs.index]
        return [cell for cell in self._adata.obs_names if cell in leaves]
