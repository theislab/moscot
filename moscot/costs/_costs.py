from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, Optional
from numbers import Number

import numpy as np
import numpy.typing as npt

from anndata import AnnData

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import networkx as nx

import numpy as np
import numpy.typing as npt

__all__ = ["LeafDistance", "BarcodeDistance"]
    

class BaseLoss(ABC):

    @abstractmethod
    def compute(self, *args: Any, **kwargs: Any) -> npt.ArrayLike:
        pass

    def __call__(self, kind: Literal["LeafDistance", "BarcodeDistance"], *args: Any, **kwargs: Any):
        if kind == "LeafDistance":
            return LeafDistance(*args, **kwargs)
        if kind == "BarcodeDistance":
            return BarcodeDistance(*args, **kwargs)

    @staticmethod
    def _normalize(cost_matrix: npt.ArrayLike, scale: Union[str, int, float] = "max") -> npt.ArrayLike:
        # TODO: @MUCDK find a way to have this for non-materialized matrices (will be backend specific)
        if scale == "max":
            cost_matrix /= cost_matrix.max()
        elif scale == "mean":
            cost_matrix /= cost_matrix.mean()
        elif scale == "median":
            cost_matrix /= np.median(cost_matrix)
        elif isinstance(scale, float):
            cost_matrix /= scale
        elif scale is None:
            pass
        else:
            raise NotImplementedError(scale)
        return cost_matrix


class BarcodeDistance(BaseLoss):
    def compute(
        self,
        adata: AnnData,
        attr: str,
        key: str,
        scale: Optional[Union[Literal["max", "mean", "median"], float]] = None,
        **_: Any,
    ) -> npt.ArrayLike:
        container = getattr(adata, attr)
        if key not in container:
            raise ValueError("TODO: no valid key")
        barcodes = container[key]
        n_cells = barcodes.shape[0]
        distances = np.zeros((n_cells, n_cells))
        for i in range(n_cells):
            print(i)
            distances[i, i + 1:] = [
                self._scaled_Hamming_distance(barcodes[i, :], barcodes[j, :]) for j in range(i + 1, n_cells)
            ]
        if scale is None:
            return distances + np.transpose(distances)
        return self._normalize(distances + np.transpose(distances), scale=scale)

    @staticmethod
    def _scaled_Hamming_distance(x: npt.ArrayLike, y: npt.ArrayLike) -> Number:
        """adapted from https://github.com/aforr/LineageOT/blob/8c66c630d61da289daa80e29061e888b1331a05a/lineageot/inference.py#L33"""

        shared_indices = (x >= 0) & (y >= 0)
        b1 = x[shared_indices]

        # There may not be any sites where both were measured
        if len(b1) == 0:
            return np.nan  # TODO: What to do if this happens?
        b2 = y[shared_indices]

        differences = b1 != b2
        double_scars = differences & (b1 != 0) & (b2 != 0)

        return (np.sum(differences) + np.sum(double_scars)) / len(b1)


class LeafDistance(BaseLoss):
    def compute(
        self,
        adata: AnnData,
        attr: str,
        key: str,
        scale: Optional[Union[Literal["max", "mean", "median"], float]] = None,
        **kwargs: Any,
    ) -> npt.ArrayLike:
        """
        Computes the matrix of pairwise distances between leaves of the tree
        """
        container = getattr(adata, attr)
        tree = container[key]
        if scale is None:
            return self._create_cost_from_tree(tree, adata, **kwargs)
        return self._normalize(
            self._create_cost_from_tree(tree, adata, **kwargs), scale
        )  # Can we do this in a stateless class?

    def _create_cost_from_tree(self, tree: nx.DiGraph, adata: AnnData, **kwargs: Any) -> npt.ArrayLike:
        # TODO(@MUCDK): make it more efficient, current problem: `target`in `multi_source_dijkstra` cannot be chosen as a subset
        undirected_tree = tree.to_undirected()
        leaves = self._get_leaves(undirected_tree, adata)
        n_leaves = len(leaves)
        distances = np.zeros((n_leaves, n_leaves))
        for i, leaf in enumerate(leaves):
            distance_dictionary = nx.multi_source_dijkstra(undirected_tree, [leaf], **kwargs)[0]
            distances[i, :] = [distance_dictionary.get(leaf) for leaf in leaves]
        return distances

    @staticmethod
    def _get_leaves(tree: nx.Graph, adata: AnnData, cell_to_leaf: Optional[Dict] = None) -> List:
        leaves = [node for node in tree if tree.degree(node) == 1]
        if not set(adata.obs.index).issubset(leaves):
            if cell_to_leaf is None:
                raise ValueError(
                    "TODO: The node names do not correspond to the anndata obs index names. Please provide a `cell_to_lead` dict."
                )
            leaves = [cell_to_leaf[cell] for cell in list(adata.obs.index)]
        return list(adata.obs.index)
