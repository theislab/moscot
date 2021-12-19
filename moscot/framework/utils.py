from typing import Any, List, Tuple, Union

from ott.geometry.costs import CostFn, Euclidean
from ott.geometry.pointcloud import PointCloud
from ott.core.gromov_wasserstein import GWLoss
from typing import Any, Dict, List, Tuple, Union, Optional, Literal
from numbers import Number

from networkx import DiGraph

from jax import numpy as jnp
from ott.geometry.costs import CostFn
from ott.geometry.geometry import Geometry
from ott.core.gromov_wasserstein import GWLoss
import numpy as np
CostFn_t = Union[CostFn, GWLoss]

from ott.geometry.costs import Euclidean

from anndata import AnnData

from moscot.framework.estimators import strategies_MatchingEstimator


def _verify_key(adata: AnnData, key: str, policy: Union[Tuple, List[Tuple], strategies_MatchingEstimator]) -> List:
    if key not in adata.obs.columns:
        raise ValueError(f"Key {key} not found in adata.obs.columns")

    if type(policy) == tuple:
        values_adata = set(adata.obs[key].values)
        for item in policy:
            if item not in values_adata:
                raise ValueError(f"Value {item} in column {key} of the AnnData object does not exist.")
        return policy

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

    else:
        raise NotImplementedError


def _prepare_geometry(
    adata: AnnData, key: str, transport_sets: List, cost_fn: Union[CostFn, None] = Euclidean, **kwargs: Any
) -> PointCloud:

    return PointCloud(
        adata[adata.obs[key] == transport_sets[0]],
        adata[adata.obs[key] == transport_sets[1]],
        cost_fn=cost_fn,
        **kwargs,
    )


def _prepare_geometries(
    adata: AnnData, key: str, transport_sets: List, cost_fn: Union[CostFn, None] = Euclidean, **kwargs: Any
) -> List[PointCloud]:

    list_geometries = []
    for tup in transport_sets:
        list_geometries.append(_prepare_geometry(adata, key, tup, cost_fn, **kwargs))

    return list_geometries


def _check_arguments(a: Optional[Union[jnp.array, List[jnp.array]]] = None,
                     b: Optional[Union[jnp.array, List[jnp.array]]] = None,
                     geometries: List[Geometry] = None):
    if (a == a and b != b) or (a != a and b == b):
        raise ValueError("Either both a and b must be provided or none of them.")
    if type(a) is list:
        if len(a) != len(b):
            raise ValueError("a and b must have the same length.")
        if len(a) != len(geometries):
            raise ValueError("a and b must have the same length as self.geometries.")

def _create_constant_weights(geometry: Geometry):

    return jnp.full(geometry.cost_matrix.shape[1], 1 / geometry.cost_matrix.shape[1]), jnp.full(geometry.cost_matrix.shape[0], 1 / geometry.cost_matrix.shape[0])