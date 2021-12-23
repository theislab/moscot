import networkx as nx
from networkx import DiGraph
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Union, Literal, Optional
from numbers import Number
import jax.numpy as jnp
from networkx.algorithms.lowest_common_ancestors import all_pairs_lowest_common_ancestor


def lca_cost(trees: Dict[int, DiGraph]) -> Dict[int, jnp.ndarray]: #TODO: specify Any, i.e. which data type trees have
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
    cost_matrix_dict = {}
    for key, tree in trees.items():
        n_nodes = len(tree.nodes)
        cost = np.zeros((n_nodes, n_nodes))
        for nodes, an in all_pairs_lowest_common_ancestor(tree): #TODO: rewrite to make it more efficient
            cost[int(nodes[0]), int(nodes[1])] = nx.dijkstra_path_length(tree, an,
                                nodes[0]) + nx.dijkstra_path_length(tree, an, nodes[1])
        cost_matrix_dict[key] = jnp.array(cost + jnp.transpose(cost))

    return cost_matrix_dict




