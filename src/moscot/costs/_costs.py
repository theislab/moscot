from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import networkx as nx
import numpy as np

from anndata import AnnData

from moscot._logging import logger
from moscot._types import ArrayLike
from moscot.base.cost import BaseCost
from moscot.costs._utils import register_cost

__all__ = ["LeafDistance", "BarcodeDistance"]


@register_cost("barcode_distance", backend="moscot")
class BarcodeDistance(BaseCost):
    """Scaled `Hamming distance <https://en.wikipedia.org/wiki/Hamming_distance>`_ between barcodes.

    .. seealso::
        - See :doc:`../notebooks/examples/problems/700_barcode_distance` on how to use this cost
          in the :class:`~moscot.problems.time.LineageProblem`.

    Parameters
    ----------
    adata
        Annotated data object.
    kwargs
        Additional keyword arguments for the :class:`~moscot.base.cost.BaseCost`.
    """

    def __init__(self, adata: AnnData, **kwargs: Any):
        super().__init__(adata, **kwargs)
        try:
            self._barcodes = getattr(self.adata, self._attr)[self._key].astype(np.int64)
        except AttributeError:
            raise AttributeError(f"`Anndata` has no attribute `{self._attr}`.") from None
        except KeyError:
            raise KeyError(f"Unable to find data in `adata.{self._attr}[{self._key!r}]`.") from None

    def _compute(
        self,
        *_: Any,
        **__: Any,
    ) -> ArrayLike:
        logger.info("Computing barcode distance")
        n_cells = self.barcodes.shape[0]

        # TODO(michalk8): use numba
        distances = np.zeros((n_cells, n_cells))
        for i in range(n_cells):
            distances[i, i + 1 :] = [
                _scaled_hamming_dist(self.barcodes[i, :], self.barcodes[j, :]) for j in range(i + 1, n_cells)
            ]
        return distances + distances.T

    @property
    def barcodes(self) -> ArrayLike:
        """Barcodes."""
        return self._barcodes


@register_cost("leaf_distance", backend="moscot")
class LeafDistance(BaseCost):
    """`Shortest path <https://en.wikipedia.org/wiki/Shortest_path_problem>`_ distance on a weighted tree.

    .. seealso::
        - See :doc:`../notebooks/examples/problems/600_leaf_distance` on how to use this cost
          in the :class:`~moscot.problems.time.LineageProblem`.

    Parameters
    ----------
    adata
        Annotated data object. The tree is always extracted from the :attr:`~anndata.AnnData.uns` attribute.
    weight
        If a :class:`str`, it is the edge weight attribute of the :attr:`tree`.
        If a function, it must accept arguments as described in
        :func:`~networkx.algorithms.shortest_paths.weighted.multi_source_dijkstra`.
    kwargs
        Keyword arguments for the :class:`~moscot.base.cost.BaseCost`.
    """

    def __init__(
        self, adata: AnnData, weight: Union[str, Callable[[Any, Any, Dict[Any, Any]], float]] = "weight", **kwargs: Any
    ):
        kwargs["attr"] = "uns"
        super().__init__(adata, **kwargs)
        self._weight = weight

        location = f"adata.{self._attr}[{self._key!r}][{self._dist_key!r}]"
        try:
            self._tree = getattr(self.adata, self._attr)[self._key][self._dist_key]
            if not isinstance(self.tree, nx.Graph):
                raise TypeError(f"Expected tree in `{location}` to be a `networkx.Graph`, found `{type(self.tree)}`.")
        except KeyError:
            raise KeyError(f"Unable to find tree in `{location}`.") from None

    def _compute(
        self,
        **kwargs: Any,
    ) -> ArrayLike:
        logger.info("Computing tree distance")
        undirected_tree = self.tree.to_undirected()
        leaves = self._get_leaves()
        distances = np.zeros((len(leaves), len(leaves)), dtype=float)

        for i, leaf in enumerate(leaves):
            # TODO(@MUCDK): more efficient, problem: `target`in `multi_source_dijkstra` cannot be chosen as a subset
            dist, _ = nx.multi_source_dijkstra(undirected_tree, [leaf], weight=self._weight, **kwargs)
            distances[i, :] = [dist.get(leaf) for leaf in leaves]

        return distances

    def _get_leaves(self, cell_to_leaf: Optional[Mapping[str, Any]] = None) -> List[Any]:
        leaves = {node for node in self.tree if self.tree.degree(node) == 1}
        if not set(self.adata.obs_names).issubset(leaves):
            if cell_to_leaf is None:
                raise ValueError("Leaves do not match `AnnData`'s observation names, please specify `cell_to_leaf`.")
            return [cell_to_leaf[cell] for cell in self.adata.obs_names]
        return [cell for cell in self.adata.obs_names if cell in leaves]

    @property
    def tree(self) -> nx.DiGraph:
        """Tree."""
        return self._tree


def _scaled_hamming_dist(x: ArrayLike, y: ArrayLike) -> float:
    # Adapted from `LineageOT <https://github.com/aforr/LineageOT/>`_.
    shared_indices = (x >= 0) & (y >= 0)
    b1 = x[shared_indices]

    # there may not be any sites where both were measured
    if not len(b1):
        return np.nan
    b2 = y[shared_indices]

    differences = b1 != b2
    double_scars = differences & (b1 != 0) & (b2 != 0)

    return (np.sum(differences) + np.sum(double_scars)) / len(b1)
