import networkx as nx
from networkx import DiGraph
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Union, Literal, Optional
from numbers import Number
import jax.numpy as jnp
from networkx.algorithms.lowest_common_ancestors import all_pairs_lowest_common_ancestor
from anndata import AnnData
from lineageot.inference import compute_tree_distances
import abc
from lineageot.inference import get_leaves

class TreeCostFn(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def compute_distance(self, tree):
        pass


class Leaf_distance(TreeCostFn):
    @staticmethod
    def compute_distance(tree) -> jnp.ndarray:
        """
        Computes the matrix of pairwise distances between leaves of the tree
        """
        return _compute_tree_distances(tree)

class LCA_distance(TreeCostFn):
    def compute_distance(self, tree) -> jnp.ndarray: #TODO: specify Any, i.e. which data type trees have
        """
        creates cost matrix from trees based on nx.algorithms.lowest_common_ancestors

        Parameters
        ----------
        trees
            Dictionary of trees for each of which a cost matrix is calculated

        Returns
            Dictionary with keys being the input keys and values being the calculated cost matrices
        -------

        """
        #cost_matrix_dict = {}
        #for key, tree in trees.items():
        #    n_nodes = len(tree.nodes)
        n_nodes = len(tree.nodes)
        cost = np.zeros((n_nodes, n_nodes))
        for nodes, an in all_pairs_lowest_common_ancestor(tree): #TODO: rewrite to make it more efficient
            cost[int(nodes[0]), int(nodes[1])] = nx.dijkstra_path_length(tree, an,
                                nodes[0]) + nx.dijkstra_path_length(tree, an, nodes[1])
        #cost_matrix_dict[key] = jnp.array(cost + jnp.transpose(cost))

        return cost

def _create_column_from_trees(adata: AnnData, new_col_name: str, trees, key):
    for val in adata.obs[key].unique():
        adata.obs[new_col_name] = np.nan
        cells = (adata.obs[key] == val)
        cells_idx = np.where(adata.obs['dpi'] == val)[0]
        cells_annot = []
        cells_annot_idx = []
        for i, obsi in enumerate(adata[cells].obs_names):
            if obsi in trees[val].nodes:
                cells_annot.append(obsi)
                cells_annot_idx.append(cells_idx[i])

        for i, obsi in enumerate(cells_annot):
            adata.obs.loc[[obsi], new_col_name] = [int(n) for n in trees[val].pred[obsi]][0]

    return adata


def _cell_cost_from_matrix(adata: AnnData, key: str, key_value:Union[str, int], pre_cost: jnp.ndarray, distance_key: str = "core") -> jnp.ndarray: #TODO: define key_value dtype better
    adata_filtered = adata[adata.obs[key] == key_value][[key, distance_key]].copy()
    n_cells = adata_filtered.shape[0]
    C = np.zeros((n_cells, n_cells))
    for i, ci in enumerate(adata_filtered.obs_names):
        C[i, :] = pre_cost[adata_filtered.obs.loc[ci, distance_key].values, adata_filtered.loc[adata_filtered.obs_names, distance_key]]
    return C


def _cell_costs_from_matrix(adata: AnnData, key: str, pre_costs_dict: Dict[int, jnp.ndarray]) -> Dict[int, jnp.ndarray]:
    """
    creates (n_cells, n_cells) cost matrix from smaller cost matrix #TODO: describe better
    Parameters
    ----------
    pre_costs_dict

    Returns
    -------

    """
    cell_cost_matrices = {}
    for tup, pre_cost in pre_costs_dict.items():
        cell_cost_matrices[tup] = _cell_cost_from_matrix(adata, key, tup, pre_cost)

def _compute_tree_distances(tree): # TODO: this is adapted from lineageOT, we want to make it more efficient.
    """
    Computes the matrix of pairwise distances between leaves of the tree
    """
    leaves = get_leaves(tree)
    num_leaves = len(leaves) - 1
    distances = np.zeros([num_leaves, num_leaves])
    for leaf_index in range(num_leaves):
        distance_dictionary, tmp = nx.multi_source_dijkstra(tree.to_undirected(), [leaves[leaf_index]], weight = 'time')
        for target_leaf_index in range(num_leaves):
            distances[leaf_index, target_leaf_index] = distance_dictionary[leaves[target_leaf_index]]
    return distances


