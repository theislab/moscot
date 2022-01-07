from typing import List, Optional, Tuple, Union, Dict

from jax import numpy as jnp
from ott.geometry.costs import CostFn
from ott.core.gromov_wasserstein import GWLoss

CostFn_t = Union[CostFn, GWLoss]
from anndata import AnnData
from moscot.framework.settings import strategies_MatchingEstimator
from moscot.framework.utils.custom_costs import Leaf_distance
from moscot.framework.geom.geometry import Geom

CostFn_t = Union[CostFn, GWLoss]
CostFn_tree = Union[Leaf_distance]
CostFn_general = Union[CostFn_t, CostFn_tree]
Scales = Union["mean", "median", "max"]


def _verify_key(adata: AnnData,
                key: str,
                policy: Union[List[Tuple], strategies_MatchingEstimator],
                subset: List = None) -> List[Tuple]:
    """
    verifies that key is a valid adata.obs column name and that the policy is actionable given the data

    Parameters
    ----------
    adata
        AnnData object containing the gene expression
    key
        column of AnnData.obs containing assignment of data points to distributions
    policy
        2-tuples of values of self.key defining the distribution which the optimal transport maps are calculated for

    Returns
        List of 2-tuples defining the (source, target) pairs of distributions which the transport maps are calculated for
    -------

    """
    if key not in adata.obs.columns:
        raise ValueError(f"Key {key} not found in adata.obs.columns")

    if isinstance(policy, list):
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
        if subset is not None:
            values_adata = values_adata.intersection(set(subset))
        sorted_values = sorted(list(values_adata))
        return [(el_1, el_2) for el_1 in sorted_values for el_2 in sorted_values if el_1 != el_2]

    elif policy == "sequential":
        values_adata = set(adata.obs[key].values)
        if subset is not None:
            values_adata = values_adata.intersection(set(subset))
        sorted_values = sorted(list(values_adata))
        return [(sorted_values[i], sorted_values[i+1]) for i in range(len(values_adata)-1)]

    else:
        raise NotImplementedError


def _check_arguments(
    a: Optional[Union[jnp.array, List[jnp.array]]],
    b: Optional[Union[jnp.array, List[jnp.array]]],
    geometry_dict: Dict[Tuple, Geom],
):
    if (a == a and b != b) or (a != a and b == b):
        raise ValueError("Either both a and b must be provided or none of them.")
    if type(a) is list:
        if len(a) != len(b):
            raise ValueError("a and b must have the same length.")
        if len(a) != len(geometry_dict):
            raise ValueError("a and b must have the same length as self.geometries.")


def _create_constant_weights_source(geometry: Geom) -> jnp.ndarray:
    num_a, _ = geometry.shape
    return jnp.ones((num_a,)) / num_a


def _create_constant_weights_target(geometry: Geom) -> jnp.ndarray:
    _, num_b = geometry.shape
    return jnp.ones((num_b,)) / num_b

def get_param_dict(param, tuple_keys):
    if isinstance(param, list):
        if len(param) != len(tuple_keys):
            raise ValueError("If 'param' is a list its length must be equal to the number of OT problems solved, "
                             "i.e. {}".format(len(tuple_keys)))
        return {tup: param[i] for i, tup in enumerate(tuple_keys)}
    elif isinstance(param, dict):
        if not bool(param):
            return {tup: param for tup in tuple_keys}
        if set(param.keys()) != set(tuple_keys):
            raise ValueError("The keys in the dictionary provided are not the same ones as expected.")
        return param
    else:
        return {tup: param for tup in tuple_keys}


def _compute_tree_cost(tree, TreeCostFn):
    return TreeCostFn.compute_distance(tree)