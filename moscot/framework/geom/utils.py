"""
Here we need registry due to moscot.framework.geom.geometry.Geom class
"""
from typing import Any, Dict, List, Tuple, Union, Optional
import logging

from scipy.sparse import issparse
import networkx as nx

from jax import numpy as jnp
from ott.geometry.costs import CostFn, Euclidean
from ott.core.gromov_wasserstein import GWLoss
import numpy as np

from anndata import AnnData

from moscot.framework.utils.utils import CostFn_tree, _compute_tree_cost
from moscot.framework.geom.geometry import Geom

CostFn_t = Union[CostFn, GWLoss]
from anndata import AnnData

from moscot.framework.geom.geometry import Geom
from moscot.framework.utils.custom_costs import Leaf_distance

CostFn_t = Union[CostFn, GWLoss]
CostFn_tree = Union[Leaf_distance]
CostFn_general = Union[CostFn_t, CostFn_tree]
Scales = Union["mean", "median", "max"]


def _prepare_xy_geometry(
    adata: AnnData,
    key: str,
    transport_tuple: Tuple,
    rep: str,
    online: bool = False,
    cost_fn: Union[CostFn, None] = Euclidean(),
    **kwargs: Any,
) -> Geom:
    """

    Parameters
    ----------
    adata
        AnnData object containing the gene expression
    key
        column of AnnData.obs containing assignment of data points to distributions
    transport_tuple
        2-Tuple defining distributions which the transport map is to be calculated for
    rep
        instance defining how the gene expression is saved in the AnnData object
    online #TODO: clarify if we want to have this as kwargs
        whether to save the geometry object online; reduces memory consumption
    cost_fn:
        Cost function to use.
    kwargs
        kwargs for Geom

    Returns
        Geom
    -------

    """
    if issparse(getattr(adata, rep)):
        return Geom(
            getattr(adata[adata.obs[key] == transport_tuple[0]], rep).todense(),
            getattr(adata[adata.obs[key] == transport_tuple[1]], rep).todense(),
            cost_fn=cost_fn,
            online=online,
            **kwargs,
        )
    else:
        return Geom(
            getattr(
                adata[adata.obs[key] == transport_tuple[0]], rep
            ),  # TODO: do we also want to allow layers, wouldn't be possible to fetch with getattr
            getattr(adata[adata.obs[key] == transport_tuple[1]], rep),
            cost_fn=cost_fn,
            online=online,
            **kwargs,
        )


def _prepare_xy_geometries(
    adata: AnnData,
    key: str,
    transport_sets: List[Tuple],
    rep: str,
    online: bool = False,  # TODO: discuss whether we want to have it as kwarg or not
    cost_fn: Union[CostFn, None] = Euclidean(),
    custom_cost_matrix_dict: Optional[Dict[Tuple, jnp.ndarray]] = None,
    scale: Optional[str] = None,
    **kwargs: Any,
) -> Dict[Tuple, Geom]:
    """

    Parameters
    ----------
    adata
        AnnData object containing the gene expression
    key
        column of AnnData.obs containing assignment of data points to distributions
    transport_sets

    rep
        instance defining how the gene expression is saved in the AnnData object
    online #TODO: clarify if we want to have this as kwargs
        whether to save the geometry object online; reduces memory consumption
    cost_fn:
        Cost function to use.
    kwargs
        kwargs for Geom

    Returns
        Dictionary with keys the tuples defining the transport and values being the corresponding moscot.framework.geometry.Geom
    -------

    """
    dict_geometries = {}
    if custom_cost_matrix_dict is None:
        for tup in transport_sets:
            dict_geometries[tup] = _prepare_xy_geometry(adata, key, tup, rep, online, cost_fn, **kwargs)
    else:
        dict_geometries = _prepare_geometries_from_cost(custom_cost_matrix_dict, scale=scale)
    return dict_geometries


def _prepare_xx_geometry(
    adata: AnnData,
    key: str,
    transport_point: Any,
    rep: str,
    online: bool = False,
    cost_fn: Union[CostFn, None] = Euclidean(),
    **kwargs: Any,
) -> Geom:
    """

    Parameters
    ----------
    adata
        AnnData object containing the gene expression
    key
        column of AnnData.obs containing assignment of data points to distributions
    transport_tuple
        2-Tuple defining distributions which the transport map is to be calculated for
    rep
        instance defining how the gene expression is saved in the AnnData object
    online #TODO: clarify if we want to have this as kwargs
        whether to save the geometry object online; reduces memory consumption
    cost_fn:
        Cost function to use.
    kwargs
        kwargs for Geom

    Returns
        Geom
    -------

    """
    if issparse(getattr(adata, rep)):
        return Geom(
            getattr(adata[adata.obs[key] == transport_point], rep).todense(),
            cost_fn=cost_fn,
            online=online,
            **kwargs,
        )
    else:
        return Geom(
            getattr(
                adata[adata.obs[key] == transport_point], rep
            ),  # TODO: do we also want to allow layers, wouldn't be possible to fetch with getattr
            cost_fn=cost_fn,
            online=online,
            **kwargs,
        )


def _prepare_xx_geometries(
    adata: AnnData,
    key: str,
    transport_points: List[Any],
    rep: str,
    online: bool = False,  # TODO: discuss whether we want to have it as kwarg or not
    cost_fn: Union[CostFn, None] = Euclidean(),
    custom_cost_matrix_dict: Optional[Dict[Any, jnp.ndarray]] = None,
    scale: Optional[str] = None,
    **kwargs: Any,
) -> Dict[Tuple, Geom]:
    """

    Parameters
    ----------
    adata
        AnnData object containing the gene expression
    key
        column of AnnData.obs containing assignment of data points to distributions
    transport_points
        subset of key.values defining the data points belonging to one distribution
    rep
        instance defining how the gene expression is saved in the AnnData object
    online #TODO: clarify if we want to have this as kwargs
        whether to save the geometry object online; reduces memory consumption
    cost_fn:
        Cost function to use.
    kwargs
        kwargs for Geom

    Returns
        Dictionary with keys the tuples defining the transport and values being the corresponding moscot.framework.geometry.Geom
    -------

    """
    dict_geometries = {}
    if custom_cost_matrix_dict is None:
        for point in transport_points:
            dict_geometries[point] = _prepare_xx_geometry(adata, key, point, rep, online, cost_fn, **kwargs)
    else:
        dict_geometries = _prepare_geometries_from_cost(custom_cost_matrix_dict, scale=scale)
    return dict_geometries


def _prepare_geometries_from_cost(
    cost_matrices_dict: Dict[Union[int, Tuple], jnp.ndarray], scale: Optional[str] = "max", **kwargs: Any
) -> Dict[Tuple, Geom]:
    dict_geometries = {}
    for key, cost_matrix in cost_matrices_dict.items():
        dict_geometries[key] = _prepare_geometry_from_cost(cost_matrix, scale, **kwargs)
    return dict_geometries


def _prepare_geometry_from_cost(cost_matrix: jnp.ndarray, scale: Optional[str] = "max", **kwargs: Any) -> Geom:
    # TODO: @MUCDK implementation of this for "online" saved matrices
    if scale == "max":
        cost_matrix /= cost_matrix.max()
    elif scale == "mean":
        cost_matrix /= cost_matrix.mean()
    elif scale == "median":
        cost_matrix /= np.median(cost_matrix)  # https://github.com/google/jax/issues/4379
    elif scale is None:
        pass
    else:
        raise NotImplementedError(scale)
    return Geom(cost_matrix=cost_matrix, **kwargs)


def _prepare_geoms_from_tree(
    tree_dict: Dict[Any, nx.DiGraph],
    TreeCostFn: Union[CostFn_tree, Dict[Any, CostFn_tree]],
    custom_cost_matrix_dict: Optional[Dict[Tuple, jnp.ndarray]] = None,
    scale: Optional[str] = None,
    **kwargs: Any,
) -> Dict[Tuple, Geom]:
    dict_geometries = {}
    if custom_cost_matrix_dict is None:
        logging.info("Calculating cost matrices ...")
        for key, tree in tree_dict.items():
            dict_geometries[key] = _prepare_geom_from_tree(tree, TreeCostFn, scale)
        logging.info("Done.")
    else:
        dict_geometries = _prepare_geometries_from_cost(custom_cost_matrix_dict, scale=scale)
    return dict_geometries


def _prepare_geom_from_tree(tree: nx.DiGraph, TreeCostFn: CostFn_tree, scale: Optional[Scales] = None) -> Geom:
    return _prepare_geometry_from_cost(_compute_tree_cost(tree, TreeCostFn), scale=scale)
