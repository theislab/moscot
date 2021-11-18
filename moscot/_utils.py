from typing import Any, Dict, Tuple, Union, Callable, Optional

import pandas as pd
import networkx as nx

import numpy as np

__all__ = ("tree_distances",)


def tree_distances(
    G: nx.DiGraph, weight: Optional[Union[str, Callable[[Tuple[Any, Any, Dict[Any, Any]]], float]]] = "length"
) -> pd.DataFrame:
    """
    Compute distance matrix from a tree.

    Parameters
    ----------
    G
        The tree.
    weight
        Edge attribute used to specify edge length. If `None`, all edges are of length `1`.

    Returns
    -------
    The distance matrix.

    References
    ----------
    Taken from `LineageOt <https://github.com/aforr/LineageOT/blob/master/lineageot/inference.py#L1154-L1164>`_.
    """
    if not isinstance(G, nx.DiGraph):
        raise TypeError(f"Expected graph to be directed, found `{type(G).__name__}`.")

    leaves = [n for (n, degree) in G.out_degree(G.nodes) if not degree]
    dist = np.empty((len(leaves), len(leaves)), dtype=np.float64)
    G_undirected = G.to_undirected()

    # TODO(michalk8): really inefficient impl. (traverses a node more than once) + not exploiting symmetry
    # in the future, traverse the tree from leaves (essentially fill-in upper diagonal, starting from diag) + T
    for i, leaf in enumerate(leaves):
        length, _ = nx.multi_source_dijkstra(G_undirected, [leaf], weight=weight)
        for j, target_leaf in enumerate(leaves):
            dist[i, j] = length[target_leaf]

    return pd.DataFrame(dist, index=leaves, columns=leaves)
