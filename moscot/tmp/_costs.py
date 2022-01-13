from typing import Union, Literal, Optional, Any
from lineageot.inference import get_leaves
import networkx as nx
import numpy as np
from numpy.typing import ArrayLike
from abc import abstractmethod, ABC


class BaseLoss(ABC):

    @abstractmethod
    def compute(self, *args: Any, **kwargs: Any) -> ArrayLike:
        pass

    def __call__(self,
                 kind: Literal,
                 scale: Optional[str] = None,
                 *args: Any,
                 **kwargs):
        if kind == "LeafDistance":
            tree = kwargs.pop("tree", None)
            return LeafDistance(tree, scale, **kwargs)
        else:
            raise NotImplementedError

    @staticmethod
    def normalize(cost_matrix: ArrayLike,
                  scale: Optional[str] = "max",
                  **kwargs: Any) -> ArrayLike:
        # TODO: @MUCDK find a way to have this for non-materialized matrices (will be backend specific)
        if scale == "max":
            cost_matrix /= cost_matrix.max()
        elif scale == "mean":
            cost_matrix /= cost_matrix.mean()
        elif scale == "median":
            cost_matrix /= np.median(cost_matrix)
        elif scale is None:
            pass
        else:
            raise NotImplementedError(scale)
        return cost_matrix


class LeafDistance(BaseLoss):

    def compute(self,
                 tree: nx.DiGraph,
                 scale: Literal,
                 **kwargs: Any):
        """
        Computes the matrix of pairwise distances between leaves of the tree
        """
        if tree is None:
            raise ValueError("For computing the LeafDistance a tree needs to be provided.")
        if scale is None:
            return _compute_leaf_distances(tree)
        else:
            return self.normalize(_compute_leaf_distances(tree), scale) #Can we do this in a stateless class?


def _compute_leaf_distances(tree): # TODO(MUCDK): this is adapted from lineageOT, we want to make it more efficient.
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


