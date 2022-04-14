from typing import Any, List, Optional

import networkx as nx

import numpy as np
import numpy.typing as npt

from anndata import AnnData

from moscot.problems import MultiMarginalProblem
from moscot.solvers._output import MatrixSolverOutput


class TestSolverOutput(MatrixSolverOutput):
    @property
    def cost(self) -> float:
        return 0.5

    @property
    def converged(self) -> bool:
        return True

    def _ones(self, n: int) -> npt.ArrayLike:
        return np.ones(n)


class MockMultiMarginalProblem(MultiMarginalProblem):
    def _estimate_marginals(self, adata: AnnData, *, source: bool, **kwargs: Any) -> Optional[npt.ArrayLike]:
        pass


def _get_random_trees(
    n_leaves: int, n_trees: int, n_initial_nodes: int = 50, leaf_names: Optional[List[List[str]]] = None, seed: int = 42
):
    rng = np.random.RandomState(42)
    if leaf_names is not None:
        assert len(leaf_names) == n_trees
    trees = []
    for tree_idx in range(n_trees):
        G = nx.random_tree(n_initial_nodes, seed=seed, create_using=nx.DiGraph)
        leaves = [x for x in G.nodes() if G.out_degree(x) == 0 and G.in_degree(x) == 1]
        inner_nodes = list(set(G.nodes()) - set(leaves))
        leaves_updated = leaves
        for i in range(n_leaves - len(leaves)):
            G.add_node(n_initial_nodes + i)
            G.add_edge(rng.choice(inner_nodes, 1)[0], n_initial_nodes + i)
            leaves_updated.append(n_initial_nodes + i)
        assert len(leaves_updated) == n_leaves
        relabel_dict = {leaves_updated[i]: leaf_names[tree_idx][i] for i in range(len(leaves_updated))}
        G = nx.relabel_nodes(G, relabel_dict)
        trees.append(G)

    return trees
