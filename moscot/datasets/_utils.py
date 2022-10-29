from typing import List, Optional

from scipy.sparse import dok_matrix
import networkx as nx

import numpy as np

from anndata import AnnData

from moscot._types import ArrayLike


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


def cost_to_obsp(
    adata: AnnData, cost_matrix: ArrayLike, obs_names_source: List[str], obs_names_target: List[str], key: str
) -> AnnData:
    """
    Add cost_matrix to :attr:`adata.obsp`.

    Parameters
    ----------
    %(adata)s

    cost_matrix
        A cost matrix to be saved to the :class:`anndata.AnnData` instance.
    obs_names_source
        List of indices corresponding to the rows of the cost matrix
    obs_names_target
        List of indices corresponding to the columns of the cost matrix
    key
        Key of :attr:`anndata.AnnData.obsp` to inscribe or add the `cost_matrix` to.

    Returns
    -------
    :class:`anndata.AnnData` with modified/added :attr:`anndata.AnnData.obsp`.
    """
    obs_names = adata.obs_names
    if not set(obs_names_source).issubset(set(obs_names)):
        raise ValueError("TODO. `obs_names_source` is not a subset of `adata.obs_names`.")
    if not set(obs_names_target).issubset(set(obs_names)):
        raise ValueError("TODO. `obs_names_target` is not a subset of `adata.obs_names`.")
    obs_names_intersection = set(obs_names_source).intersection(set(obs_names_target))
    if len(obs_names_intersection) and set(obs_names_source) != set(obs_names_target):
        raise ValueError(
            "TODO. `obs_names_source` and `obs_names_target` must contain the same values or have empty intersection."
        )
    other_obs_names = set(obs_names) - set(obs_names_source) - set(obs_names_target)
    adata_reordered = adata[list(obs_names_source) + list(obs_names_target) + list(other_obs_names)]
    if key in adata.obsp:
        obsp_layer = adata_reordered.obsp[key]
    else:
        obsp_layer = dok_matrix((adata.n_obs, adata.n_obs))
    col_ilocs = (
        range(len(obs_names_target))
        if len(obs_names_intersection)
        else range(len(obs_names_source), len(obs_names_source) + len(obs_names_target))
    )
    obsp_layer.obsp[0 : len(obs_names_source), col_ilocs] = cost_matrix
    adata_reordered.obsp[key] = obsp_layer
    return adata[obs_names]
