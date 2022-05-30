from typing import Any, List, Type, Tuple, Union, Optional

import networkx as nx

import numpy as np

from anndata import AnnData

from moscot._types import ArrayLike
from moscot.problems.base import OTProblem, CompoundProblem, MultiMarginalProblem
from moscot.solvers._output import MatrixSolverOutput
from moscot.problems.base._mixins import AnalysisMixin
from moscot.problems.base._compound_problem import B

Geom_t = Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
RTOL = 1e-6
ATOL = 1e-6


class CompoundProblemWithMixin(CompoundProblem, AnalysisMixin):
    @property
    def _base_problem_type(self) -> Type[B]:
        return OTProblem

    @property
    def _valid_policies(self) -> Tuple[str, ...]:
        return ()


class MockSolverOutput(MatrixSolverOutput):
    @property
    def cost(self) -> float:
        return 0.5

    @property
    def converged(self) -> bool:
        return True

    def _ones(self, n: int) -> ArrayLike:
        return np.ones(n)


class MockMultiMarginalProblem(MultiMarginalProblem):
    def _estimate_marginals(self, adata: AnnData, *, source: bool, **kwargs: Any) -> Optional[ArrayLike]:
        pass


class MockBaseSolverOutput:
    def __init__(self, len_a: int, len_b: int):
        rng = np.random.RandomState(42)
        self.a = rng.randn(len_a)
        self.b = rng.randn(len_b)


def _get_random_trees(
    n_leaves: int, n_trees: int, n_initial_nodes: int = 50, leaf_names: Optional[List[List[str]]] = None, seed: int = 42
) -> Tuple[nx.DiGraph, ...]:
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
