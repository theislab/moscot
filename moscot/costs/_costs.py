from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union, Literal, Mapping, Optional

from scipy.sparse import csr_matrix
import networkx as nx

import numpy as np

from anndata import AnnData

from moscot._types import ArrayLike

__all__ = ["LeafDistance", "BarcodeDistance", "BaseLoss", "cost_to_obsp"]
Scale_t = Literal["max", "min", "median", "mean"]


class BaseLoss(ABC):
    """Base class handling all :mod:`moscot` losses."""

    @abstractmethod
    def _compute(self, *args: Any, **kwargs: Any) -> ArrayLike:
        pass

    def __init__(self, adata: AnnData, attr: str, key: str, dist_key: Union[Any, Tuple[Any, Any]]):
        self._adata = adata
        self._attr = attr
        self._key = key
        self._dist_key = dist_key

    def __call__(self, *args: Any, **kwargs: Any) -> ArrayLike:
        return self._compute(*args, **kwargs)

    @classmethod
    def create(cls, kind: Literal["leaf_distance", "barcode_distance"], *args: Any, **kwargs: Any) -> "BaseLoss":
        if kind == "leaf_distance":
            return LeafDistance(*args, **kwargs)
        if kind == "barcode_distance":
            return BarcodeDistance(*args, **kwargs)
        raise NotImplementedError(f"Cost function `{kind}` is not yet implemented.")


class BarcodeDistance(BaseLoss):
    """Class handling Barcode distances."""

    def _compute(
        self,
        *_: Any,
        **__: Any,
    ) -> ArrayLike:
        container = getattr(self._adata, self._attr)
        if self._key not in container:
            raise KeyError(f"Unable to find data in `adata.{self._attr}[{self._key!r}]`.")
        barcodes = container[self._key]
        n_cells = barcodes.shape[0]
        distances = np.zeros((n_cells, n_cells))
        for i in range(n_cells):
            distances[i, i + 1 :] = [
                self._scaled_Hamming_distance(barcodes[i, :], barcodes[j, :]) for j in range(i + 1, n_cells)
            ]
        return distances + np.transpose(distances)

    @staticmethod
    def _scaled_Hamming_distance(x: ArrayLike, y: ArrayLike) -> float:
        """
        adapted from https://github.com/aforr/LineageOT/blob/8c66c630d61da289daa80e29061e888b1331a05a/lineageot/inference.py#L33.  # noqa: E501
        """

        shared_indices = (x >= 0) & (y >= 0)
        b1 = x[shared_indices]

        # There may not be any sites where both were measured
        if not len(b1):
            # TODO(@MUCDK): What to do if this happens? set to maximum or raise, depending on what user wants
            return np.nan
        b2 = y[shared_indices]

        differences = b1 != b2
        double_scars = differences & (b1 != 0) & (b2 != 0)

        return (np.sum(differences) + np.sum(double_scars)) / len(b1)


class LeafDistance(BaseLoss):
    """Class handling leaf distances (from trees)."""

    def _compute(
        self,
        **kwargs: Any,
    ) -> ArrayLike:
        """
        Compute the matrix of pairwise distances between leaves of the tree
        """
        if self._attr == "uns":
            tree = self._adata.uns[self._key][self._dist_key]
            if not isinstance(tree, nx.DiGraph):
                raise TypeError(
                    f"Expected the tree in `adata.uns['trees'][{self._dist_key!r}]` "
                    f"to be a `networkx.DiGraph`, found `{type(tree)}`."
                )
            return self._create_cost_from_tree(tree, **kwargs)
        raise NotImplementedError(f"Extracting trees from `adata.{self._attr}` is not implemented.")

    def _create_cost_from_tree(self, tree: nx.DiGraph, **kwargs: Any) -> ArrayLike:
        # TODO(@MUCDK): more efficient, problem: `target`in `multi_source_dijkstra` cannot be chosen as a subset
        undirected_tree = tree.to_undirected()
        leaves = self._get_leaves(undirected_tree)
        n_leaves = len(leaves)
        distances = np.zeros((n_leaves, n_leaves))
        for i, leaf in enumerate(leaves):
            distance_dictionary = nx.multi_source_dijkstra(undirected_tree, [leaf], **kwargs)[0]
            distances[i, :] = [distance_dictionary.get(leaf) for leaf in leaves]
        return distances

    def _get_leaves(self, tree: nx.DiGraph, cell_to_leaf: Optional[Mapping[str, Any]] = None) -> List[Any]:
        leaves = [node for node in tree if tree.degree(node) == 1]
        if not set(self._adata.obs_names).issubset(leaves):
            if cell_to_leaf is None:
                raise ValueError("Leaves do not match `AnnData`'s observation names, please specify `cell_to_leaf`.")
            return [cell_to_leaf[cell] for cell in self._adata.obs.index]
        return [cell for cell in self._adata.obs_names if cell in leaves]


def cost_to_obsp(
    adata: AnnData,
    key: str,
    cost_matrix: ArrayLike,
    obs_names_source: List[str],
    obs_names_target: List[str],
) -> AnnData:
    """
    Add cost_matrix to :attr:`adata.obsp`.

    Parameters
    ----------
    %(adata)s

    key
        Key of :attr:`anndata.AnnData.obsp` to inscribe or add the `cost_matrix` to.
    cost_matrix
        A cost matrix to be saved to the :class:`anndata.AnnData` instance.
    obs_names_source
        List of indices corresponding to the rows of the cost matrix
    obs_names_target
        List of indices corresponding to the columns of the cost matrix

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
        obsp_layer = csr_matrix((adata.n_obs, adata.n_obs))  # lil_matrix has no view in anndata.AnnData
    col_ilocs = (
        range(len(obs_names_target))
        if len(obs_names_intersection)
        else range(len(obs_names_source), len(obs_names_source) + len(obs_names_target))
    )
    obsp_layer[0 : len(obs_names_source), col_ilocs] = cost_matrix
    adata_reordered.obsp[key] = obsp_layer
    return adata_reordered[obs_names]
