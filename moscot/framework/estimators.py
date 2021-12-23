from typing import Any, Dict, List, Tuple, Union, Literal, Optional
from numbers import Number

from networkx import DiGraph
from jax import numpy as jnp
from ott.geometry.costs import CostFn
from ott.geometry.geometry import Geometry
from ott.core.gromov_wasserstein import GWLoss
import numpy as np

CostFn_t = Union[CostFn, GWLoss]

from ott.geometry.epsilon_scheduler import Epsilon

from anndata import AnnData

from moscot._solver import FusedGW, Regularized, RegularizedOT
from moscot.framework.utils import (
    _verify_key,
    _check_arguments,
    _prepare_geometry,
    _prepare_geometries,
    _create_constant_weights_source,
    _create_constant_weights_target,
    _prepare_geometries_from_cost
)
from moscot.framework.custom_costs import lca_cost
from moscot.framework.BaseProblem import BaseProblem
from moscot.framework.settings import strategies_MatchingEstimator
from moscot.framework.results import BaseResult, OTResult


class OTEstimator(BaseProblem, RegularizedOT):
    def __init__(
        self,
        adata: AnnData,
        key: str,
        params: Dict,
        cost_fn: Optional[CostFn_t] = None,
        epsilon: Optional[Union[float, Epsilon]] = None,
        **kwargs: Any,
    ) -> None:
        """
        General estimator class whose subclasses solve specific OT problems.

        Parameters
        ----------
        adata
            AnnData object containing the gene expression for all data points
        key
            column of AnnData.obs containing assignment of data points to distributions
        params
            #TODO: clarify
        cost_fn
            Cost function to use. Default is euclidean.
        epsilon
            regularization parameter for OT problem
        kwargs:
            ott.sinkhorn.sinkhorn kwargs
        """
        super().__init__(adata=adata, key=key, cost_fn=cost_fn, epsilon=epsilon, params=params)
        self._kwargs: Dict[str, Any] = kwargs

    def serialize_to_adata(self) -> Optional[AnnData]:
        pass

    def load_from_adata(self) -> None:
        pass

    def prepare(
        self,
        key: Union[str, None],
        policy: None,
        rep: None,
        cost_fn: Union[CostFn, None],
        eps: Union[float, None],
        groups: Union[List[str], Tuple[str]],
    ) -> None:
        pass

    def estimate_growth_rates(self) -> None:
        # https://github.com/broadinstitute/wot/blob/master/wot/gene_set_scores.py
        pass


class MatchingEstimator(OTEstimator):
    """
    This estimator handles linear OT problems
    """
    def __init__(
        self,
        adata: AnnData,
        key: str,
        params: Dict = None,
        cost_fn: Optional[CostFn_t] = None,
        epsilon: Optional[Union[float, Epsilon]] = None,
        **kwargs: Any,
    ) -> None:
        """

        Parameters
        ----------
        adata
        key
        params
        cost_fn
        epsilon
        kwargs: ott.sinkhorn.sinkhorn kwargs
        """
        self.geometries_dict = None
        self.a_dict = None  # TODO: check whether we can put them in class of higher order
        self.b_dict = None
        self._solver_dict = None
        self.rep = None
        super().__init__(adata=adata, key=key, cost_fn=cost_fn, epsilon=epsilon, params=params, **kwargs)

    def prepare(
        self,
        policy: Union[Tuple, List[Tuple], strategies_MatchingEstimator],
        rep: str = "X",
        cost_fn: Union[CostFn, None] = None,
        **kwargs: Any,
    ) -> None:
        """

        Parameters
        ----------
        policy
        rep
        cost_fn
        kwargs: ott.Geometry kwargs

        Returns
        -------

        """
        self.rep = rep
        transport_sets = _verify_key(self._adata, self.key, policy)
        if not isinstance(getattr(self._adata, rep), np.ndarray):
            raise ValueError("Please provide a valid layer from the")

        self.geometries_dict = _prepare_geometries(self.adata, self.key, transport_sets, self.rep, cost_fn, **kwargs)

    def fit(
        self,
        a: Optional[Union[jnp.array, List[jnp.array]]] = None,
        b: Optional[Union[jnp.array, List[jnp.array]]] = None,
    ) -> "OTResult":
        """

        Parameters
        ----------
        a
        b

        Returns
        -------

        """

        if self.geometries_dict is None:
            raise ValueError("Please run 'prepare()' first.")

        _check_arguments(a, b, self.geometries_dict)

        if a is None: #TODO: discuss whether we want to have this here, i.e. whether we want to explicitly create weights because OTT would do this for us.
            #TODO: atm do it here to have all parameter saved in the estimator class
            self.a_dict = {tup: _create_constant_weights_source(geom) for tup, geom in self.geometries_dict.items()}
            self.b_dict = {tup: _create_constant_weights_target(geom) for tup, geom in self.geometries_dict.items()}

        self._solver_dict = {tup: Regularized(cost_fn=self.cost_fn, epsilon=self.epsilon) for tup, _ in self.geometries_dict.items()}
        for tup, geom in self.geometries_dict.items():
            self._solver_dict[tup].fit(self.geometries_dict[tup], self.a_dict[tup], self.b_dict[tup])

        return OTResult(self.adata, self.key, self._solver_dict)

    def converged(self) -> Optional[bool]:
        pass

    def matrix(self) -> jnp.ndarray:
        pass

    def transport(self, inputs: jnp.ndarray, forward: bool = True) -> jnp.ndarray:
        pass


class LineageEstimator(OTEstimator):
    """
    This estimator handles FGW estimators for temporal data
    """
    def __init__(
        self,
        adata: AnnData,
        key: str,
        trees: Dict[int, DiGraph],
        params: Dict,
        cost_fn: Optional[CostFn_t] = None,
        epsilon: Optional[Union[float, Epsilon]] = None,
        alpha: Number = 0.5, # TODO: adapt from paper
        tree_rep: Union[str, DiGraph, None] = None,
        tree_cost: Union["MLE", "edgesum", "uniform_edge"] = None,
        **kwargs: Any,
    ) -> None:
        self.tree_dict = trees
        self.alpha = alpha  # #TODO: currently all pairs are computed with the same alpha, maybe change
        self.tree_rep = tree_rep
        self.tree_cost = tree_cost
        self.geometries_inter_dict: Dict[Tuple, Geometry] = None
        self.geometries_intra_dict: Dict[int, Geometry] = None
        self.a_dict: Dict[Tuple, jnp.ndarray] = None  # TODO: check whether we can put them in class of higher order
        self.b_dict: Dict[Tuple, jnp.ndarray] = None
        self.cost_intra: Dict[int, jnp.ndarray] = None
        self._solver_dict: Dict[Tuple, FusedGW] = None
        super().__init__(adata=adata, key=key, cost_fn=cost_fn, epsilon=epsilon, params=params, **kwargs)

    def prepare(
        self,
        policy: Union[Tuple, List[Tuple]],
        rep: str = "X",
        cost_fn: Optional[CostFn] = None,  # cost function for linear problem
        **kwargs
    ) -> None:
        """

        Parameters
        ----------
        policy
        rep
        cost_fn
        kwargs: kwargs for ott.geometry

        Returns
        -------

        """
        self.rep = rep
        self._scale = kwargs.pop("scale", "max")
        if self.key not in self.adata.obs.columns:
            raise ValueError(f"The provided key {self.key} is not found in the AnnData object.")
        transport_sets = _verify_key(self._adata, self.key, policy)

        self.cost_intra_dict = lca_cost(self.tree_dict)
        self.geometries_inter_dict = _prepare_geometries(self.adata, self.key, transport_sets, self.rep, cost_fn=cost_fn, **kwargs)
        self.geometries_intra_dict = _prepare_geometries_from_cost(self.cost_intra_dict,
                                                                scale=self._scale)  # TODO: here we assume we can never save it as online=True

        #TODO: add some tests here, e.g. costs should be positive

    def fit(
            self,
            a: Optional[Union[jnp.array, List[jnp.array]]] = None,
            b: Optional[Union[jnp.array, List[jnp.array]]] = None,
    ) -> "OTResult":

        if self.geometries_inter_dict is None or self.geometries_intra_dict is None:
            raise ValueError("Please run 'prepare()' first.")

        _check_arguments(a, b, self.geometries_inter_dict)

        if a is None: #TODO: discuss whether we want to have this here, i.e. whether we want to explicitly create weights because OTT would do this for us.
            #TODO: atm do it here to have all parameter saved in the estimator class
            self.a_dict = {tup: _create_constant_weights_source(geom) for tup, geom in self.geometries_inter_dict.items()}
            self.b_dict = {tup: _create_constant_weights_target(geom) for tup, geom in self.geometries_inter_dict.items()}

        self._solver_dict = {tup: FusedGW(alpha=self.alpha, epsilon=self.epsilon) for tup, _ in self.geometries_inter_dict.items()}
        for tup, geom in self.geometries_inter_dict.items():
            self._solver_dict[tup].fit(self.geometries_inter_dict[tup], self.geometries_intra_dict[tup[0]], self.geometries_intra_dict[tup[1]], self.a_dict[tup], self.b_dict[tup])

        return OTResult(self.adata, self.key, self._solver_dict)

    def converged(self) -> Optional[bool]:
        pass

    def matrix(self) -> jnp.ndarray:
        pass

    def transport(self, inputs: jnp.ndarray, forward: bool = True) -> jnp.ndarray:
        pass


class SpatialAlignmentEstimator(OTEstimator):
    """
    This estimator ...
    """
    def __init__(
        self,
        adata: AnnData,
        key: str,
        params: Dict,
        cost_fn: Optional[CostFn_t] = None,
        epsilon: Optional[Union[float, Epsilon]] = None,
        alpha: Number = None,
        spatial_rep: Union[str, np.ndarray, None] = None,
        spatial_cost: Union["spatial_cost", np.ndarray, None] = None,  # TODO: specify spatial cost
        **kwargs: Any,
    ) -> None:

        super().__init__(adata=adata, key=key, cost_fn=cost_fn, epsilon=epsilon, params=params, **kwargs)


class SpatialMappingEstimator(OTEstimator):
    """
    This estimator ...
    """
    def __init__(
        self,
        adata: AnnData,
        key: str,
        params: Dict,
        cost_fn: Optional[CostFn_t] = None,
        epsilon: Optional[Union[float, Epsilon]] = None,
        alpha: Number = None,
        spatial_rep: Union[str, np.ndarray, None] = None,
        spatial_cost: Union["spatial_cost", np.ndarray, None] = None,  # TODO: specify spatial cost
        reference_var: Union[List, None] = None,
        **kwargs: Any,
    ) -> None:

        super().__init__(adata=adata, key=key, cost_fn=cost_fn, epsilon=epsilon, params=params, **kwargs)


