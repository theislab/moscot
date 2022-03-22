from typing import Dict, List, Optional
from numbers import Number

import networkx as nx

import numpy as np
import numpy.typing as npt

from anndata import AnnData


class CostMixin:
    def _create_cost_from_tree(self, tree: nx.DiGraph, **kwargs) -> npt.ArrayLike:
        undirected_tree = tree.to_undirected()
        leaves = self.get_leaves(undirected_tree)
        n_leaves = len(leaves)
        distances = np.zeros((n_leaves, n_leaves))
        for i, leaf in enumerate(leaves):
            distance_dictionary = nx.multi_source_dijkstra(undirected_tree, [leaf], target=[leaves[i + 1 :]])[0]
            distances[i, i + 1 :] = [distance_dictionary.get(leaf) for leaf in leaves]
        return distances + np.transpose(distances)

    def _create_cost_from_barcodes(self, barcodes: npt.ArrayLike, **kwargs) -> npt.ArrayLike:
        n_cells = barcodes.shape[0]
        distances = np.zeros((n_cells, n_cells))
        for i in range(n_cells):
            distances[i, i + 1] = [
                self._scaled_Hamming_distance(barcodes[i, :], barcodes[j, :]) for j in range(i + 1, n_cells)
            ]
        return distances + np.transpose(distances)

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

    @staticmethod
    def _get_leaves(tree: nx.Graph, adata: AnnData, cell_to_leaf: Optional[Dict] = None) -> List:
        leaves = [node for node in tree if tree.degree(node) == 1]
        if leaves != list(adata.obs.index):
            if cell_to_leaf is None:
                raise ValueError(
                    "TODO: The node names do not correspond to the anndata obs index names. Please provide a `cell_to_lead` dict."
                )
            leaves = [cell_to_leaf[cell] for cell in list(adata.obs.index)]
        return leaves
