from typing import Any, List, Optional, Tuple, Union, Dict


from jax import numpy as jnp
from ott.geometry.costs import CostFn, Euclidean
from ott.geometry.geometry import Geometry
from ott.geometry.pointcloud import PointCloud
from ott.core.gromov_wasserstein import GWLoss
import numpy as np
CostFn_t = Union[CostFn, GWLoss]

from ott.geometry.costs import Euclidean

from anndata import AnnData

from moscot.framework.settings import strategies_MatchingEstimator


def _verify_key(adata: AnnData,
                key: str,
                policy: Union[Tuple, List[Tuple],
                strategies_MatchingEstimator]) -> List:
    if key not in adata.obs.columns:
        raise ValueError(f"Key {key} not found in adata.obs.columns")

    if isinstance(policy, tuple):
        if len(policy) != 2:
            raise ValueError("The length of the tuple must be 2.")
        values_adata = set(adata.obs[key].values)
        for item in policy:
            if item not in values_adata:
                raise ValueError(f"Value {item} in column {key} of the AnnData object does not exist.")
        return [policy]

    elif type(policy) == list:
        values_adata = set(adata.obs[key].values)
        for tup in policy:
            if len(tup) != 2:
                raise ValueError("The length of the tuples must be 2.")
            for item in tup:
                if item not in values_adata:
                    raise ValueError(f"Value {item} in column {key} of the AnnData object does not exist.")
        return policy

    elif policy == "pairwise":
        values_adata = set(adata.obs[key].values)
        return [(el_1, el_2) for el_1 in sorted(values_adata) for el_2 in sorted(values_adata) if el_1 != el_2]

    elif policy == "sequential":
        values_adata = set(adata.obs[key].values)
        return [(values_adata[i], values_adata[i+1]) for i in range(len(values_adata))]

    else:
        raise NotImplementedError


def _prepare_geometry(
        adata: AnnData,
        key: str,
        transport_tuple: Tuple,
        rep: str,
        cost_fn: Union[CostFn, None] = Euclidean,
        **kwargs: Any
) -> PointCloud:

    return PointCloud(
        getattr(adata[adata.obs[key] == transport_tuple[0]], rep), #TODO: do we also want to allow layers, wouldn't be possible to fetch with getattr
        getattr(adata[adata.obs[key] == transport_tuple[1]], rep),
        cost_fn=cost_fn,
        **kwargs,
    )


def _prepare_geometries(
        adata: AnnData,
        key: str,
        transport_sets: List[Tuple],
        rep: str,
        cost_fn: Union[CostFn, None] = Euclidean,
        **kwargs: Any
) -> Dict[Tuple, PointCloud]:

    dict_geometries = {}
    for tup in transport_sets:
        dict_geometries[tup] = _prepare_geometry(adata, key, tup, cost_fn, rep, **kwargs)

    return dict_geometries


def _prepare_geometries_from_cost(cost_matrices_dict: Dict[Tuple, jnp.ndarray],
                                  scale: Optional[str] = "max",
                                  **kwargs: Any) -> Dict[Geometry]:
    dict_geometries = {}
    for tup, cost_matrix in cost_matrices_dict:
        dict_geometries[tup] = _prepare_geometry_from_cost(cost_matrix, scale, **kwargs)


def _prepare_geometry_from_cost(cost_matrix: jnp.ndarray,
                                scale: Optional[str] = "max",
                                **kwargs: Any) -> Geometry:
    if scale == "max":
        cost_matrix /= cost_matrix.max()
    elif scale == "mean":
        cost_matrix /= cost_matrix.mean()
    elif scale == "median":
        cost_matrix /= np.median(cost_matrix)  # https://github.com/google/jax/issues/4379
    else:
        raise NotImplementedError(scale)
    return Geometry(cost_matrix=cost_matrix, **kwargs)

def _check_arguments(
    a: Optional[Union[jnp.array, List[jnp.array]]],
    b: Optional[Union[jnp.array, List[jnp.array]]],
    geometry_dict: Dict[Tuple, Geometry],
):
    if (a == a and b != b) or (a != a and b == b):
        raise ValueError("Either both a and b must be provided or none of them.")
    if type(a) is list:
        if len(a) != len(b):
            raise ValueError("a and b must have the same length.")
        if len(a) != len(geometry_dict):
            raise ValueError("a and b must have the same length as self.geometries.")


def _create_constant_weights_source(geometry: Geometry) -> jnp.ndarray:
    num_a, _ = geometry.shape
    return jnp.ones((num_a,)) / num_a


def _create_constant_weights_target(geometry: Geometry) -> jnp.ndarray:
    _, num_b = geometry.shape
    return jnp.ones((num_b,)) / num_b

