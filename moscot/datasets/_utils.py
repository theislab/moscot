from typing import List, Optional

import networkx as nx

import numpy as np


def _get_random_trees(
    n_leaves: int, n_trees: int, n_initial_nodes: int = 50, leaf_names: Optional[List[List[str]]] = None, seed: int = 42
) -> List[nx.DiGraph]:
    rng = np.random.RandomState(42)
    if leaf_names is not None:

        assert len(leaf_names) == n_trees
        for i in range(n_trees):
            assert len(leaf_names[i]) == n_leaves
    trees = []
    for tree_idx in range(n_trees):
        G = nx.random_tree(n_initial_nodes, seed=seed, create_using=nx.DiGraph)
        leaves = [x for x in G.nodes() if G.out_degree(x) == 0 and G.in_degree(x) == 1]
        inner_nodes = list(set(G.nodes()) - set(leaves))
        leaves_updated = leaves.copy()
        for i in range(n_leaves - len(leaves)):
            G.add_node(n_initial_nodes + i)
            G.add_edge(rng.choice(inner_nodes, 1)[0], n_initial_nodes + i)
            leaves_updated.append(n_initial_nodes + i)
        assert len(leaves_updated) == n_leaves
        if leaf_names is not None:
            relabel_dict = {leaves_updated[i]: leaf_names[tree_idx][i] for i in range(len(leaves_updated))}
            G = nx.relabel_nodes(G, relabel_dict)
        trees.append(G)

    return trees
