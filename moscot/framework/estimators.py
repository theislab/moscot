from typing import Any, Dict, List, Tuple, Union, Literal, Optional
from numbers import Number

from networkx import DiGraph
from jax import numpy as jnp
from ott.geometry.costs import CostFn
from ott.geometry.geometry import Geometry
from ott.core.gromov_wasserstein import GWLoss
import numpy as np
from ott.geometry.costs import CostFn, Euclidean

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
    _prepare_geometries_from_cost,
    get_param_dict,
)
from moscot.framework.custom_costs import lca_cost
from moscot.framework.BaseProblem import BaseProblem
from moscot.framework.settings import strategies_MatchingEstimator
from moscot.framework.results import BaseResult, OTResult


class OTEstimator(BaseProblem):
    def __init__(
        self,
        adata: AnnData,
        rep: str = "X",
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
        rep
            instance defining how the gene expression is saved in adata
        kwargs:
            ott.sinkhorn.sinkhorn kwargs
        """
        super().__init__(adata=adata, rep=rep)
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
        rep: str = "X",
        **kwargs: Any,
    ) -> None:
        """

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
        rep
            instance defining how the gene expression is saved in adata
        kwargs:
            ott.sinkhorn.sinkhorn kwargs
        """
        self.geometries_dict: Dict[Tuple, Geometry] = {}
        self.a_dict: Dict[Tuple, Geometry] = {}  # TODO: check whether we can put them in class of higher order
        self.b_dict: Dict[Tuple, Geometry] = {}
        self._solver_dict: Dict[Tuple, Regularized] = {}
        super().__init__(adata=adata, rep=rep, **kwargs)

    def prepare(
        self,
        key: str,
        policy: Union[List[Tuple], strategies_MatchingEstimator],
        subset: List = None, # e.g. time points [1,3,5,7]
        a: Optional[Union[jnp.array, List[jnp.array]]] = None,
        b: Optional[Union[jnp.array, List[jnp.array]]] = None,
        cost_fn: Optional[CostFn_t] = Euclidean(),
        custom_cost_matrix_dict: Optional[Dict[Tuple, jnp.ndarray]] = None,
        **kwargs: Any,
    ) -> "MatchingEstimator":
        """

        Parameters
        ----------
        policy
            2-tuples of values of self.key defining the distribution which the optimal transport maps are calculated for

        a:
            weights for source distribution. If of type jnp.ndarray the same distribution is taken for all models, if of type
            List[jnp.ndarray] the length of the list must be equal to the number of transport maps defined in prepare()
        b:
            weights for target distribution. If of type jnp.ndarray the same distribution is taken for all models, if of type
            List[jnp.ndarray] the length of the list must be equal to the number of transport maps defined in prepare()
        Returns
            None
        -------
        """
        self.key = key
        self.cost_fn = cost_fn
        transport_sets = _verify_key(self._adata, self.key, policy, subset)
        if not isinstance(getattr(self._adata, self.rep), np.ndarray):
            raise ValueError("Please provide a valid layer from the")

        self.geometries_dict = _prepare_geometries(self.adata, key=self.key, transport_sets=transport_sets, rep=self.rep, cost_fn=self.cost_fn, custom_cost_matrix_dict=custom_cost_matrix_dict, **kwargs)

        _check_arguments(a, b, self.geometries_dict)

        if a is None:  # TODO: discuss whether we want to have this here, i.e. whether we want to explicitly create weights because OTT would do this for us.
            # TODO: atm do it here to have all parameter saved in the estimator class
            self.a_dict = {tup: _create_constant_weights_source(geom) for tup, geom in self.geometries_dict.items()}
            self.b_dict = {tup: _create_constant_weights_target(geom) for tup, geom in self.geometries_dict.items()}

        return self

    def fit(
        self,
        epsilon: Optional[Union[List[Union[float, Epsilon]], float, Epsilon]] = 0.5,
        tau_a: Optional[Union[List, Dict[Tuple, List], float]] = 1.0,
        tau_b: Optional[Union[List, Dict[Tuple, List], float]] = 1.0,
        sinkhorn_kwargs: Optional[Union[List, Dict[Tuple, List]]] = None,
        **kwargs
    ) -> "OTResult":
        """

        Parameters
        ----------
        sinkhorn_kwargs: estimator-specific kwargs for ott.core.sinkhorn.sinkhorn
        **kwargs: ott.core.sinkhorn.sinkhorn keyword arguments applied to all estimators
        Returns
            moscot.framework.results.OTResult
        -------

        """

        if not bool(self.geometries_dict):
            raise ValueError("Please run 'prepare()' first.")

        tuples = list(self.geometries_dict.keys())
        self.epsilon_dict = get_param_dict(epsilon, tuples)
        self.tau_a_dict = get_param_dict(tau_a, tuples)
        self.tau_b_dict = get_param_dict(tau_b, tuples)
        self.sinkorn_kwargs_dict = get_param_dict(sinkhorn_kwargs, tuples)

        self._solver_dict = {tup: Regularized(cost_fn=self.cost_fn, epsilon=self.epsilon_dict[tup]) for tup in tuples}
        for tup, geom in self.geometries_dict.items():
            self._solver_dict[tup].fit(self.geometries_dict[tup], self.a_dict[tup], self.b_dict[tup], tau_a=self.tau_a_dict[tup], tau_b=self.tau_b_dict[tup], **kwargs)

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
        rep: str = "X",
        tree_cost: Union["MLE", "edgesum", "uniform_edge"] = None,
        **kwargs: Any,
    ) -> None:
        """

        Parameters
        ----------
        adata
            AnnData object containing the gene expression for all data points
        key
            column of AnnData.obs containing assignment of data points to distributions
        trees
            dictionary with keys being the time points and values the corresponding lineage tree
        params
            #TODO: clarify
        cost_fn
            Cost function to use. Default is euclidean.
        epsilon
            regularization parameter for OT problem
        alpha
            penalty term of FGW, i.e. cost = cost(GW) + alpha * cost(linear_OT)
        rep
            instance defining how the gene expression is saved in the AnnData object
        kwargs:
            ott.sinkhorn.sinkhorn kwargs
        """
        self.tree_dict = trees
        self.alpha = alpha  # #TODO: currently all pairs are computed with the same alpha, maybe change
        self.tree_cost = tree_cost
        self.geometries_inter_dict: Dict[Tuple, Geometry] = {}
        self.geometries_intra_dict: Dict[int, Geometry] = {}
        self.a_dict: Dict[Tuple, jnp.ndarray] = {}  # TODO: check whether we can put them in class of higher order
        self.b_dict: Dict[Tuple, jnp.ndarray] = {}
        self.cost_intra: Dict[int, jnp.ndarray] = {}
        self._solver_dict: Dict[Tuple, FusedGW] = {}
        super().__init__(adata=adata, key=key, cost_fn=cost_fn, epsilon=epsilon, params=params, rep=rep, **kwargs)

    def prepare(
        self,
        policy: Union[Tuple, List[Tuple]],
        **kwargs
    ) -> None:
        """

        Parameters
        ----------
        policy
            2-tuples of values of self.key defining the distribution which the optimal transport maps are calculated for
        kwargs:
            kwargs for ott.geometry

        Returns
        -------

        """
        self._scale = kwargs.pop("scale", "max")
        if self.key not in self.adata.obs.columns:
            raise ValueError(f"The provided key {self.key} is not found in the AnnData object.")
        transport_sets = _verify_key(self._adata, self.key, policy)

        self.pre_cost_intra_dict = lca_cost(self.tree_dict)
        self.cost_intra_dict = _cell_costs_from_matrix(self.pre_cost_intra_dict)
        self.geometries_inter_dict = _prepare_geometries(self.adata, self.key, transport_sets, self.rep, cost_fn=self.cost_fn, **kwargs)
        self.geometries_intra_dict = _prepare_geometries_from_cost(self.cost_intra_dict,
                                                                scale=self._scale)  # TODO: here we assume we can never save it as online=True

        #TODO: add some tests here, e.g. costs should be positive

    def fit(
            self,
            a: Optional[Union[jnp.array, List[jnp.array]]] = None,
            b: Optional[Union[jnp.array, List[jnp.array]]] = None,
    ) -> "OTResult":
        """

        Parameters
        ----------
        a:
            weights for source distribution. If of type jnp.ndarray the same distribution is taken for all models, if of type
            List[jnp.ndarray] the length of the list must be equal to the number of transport maps defined in prepare()
        b:
            weights for target distribution. If of type jnp.ndarray the same distribution is taken for all models, if of type
            List[jnp.ndarray] the length of the list must be equal to the number of transport maps defined in prepare()

        Returns
            moscot.framework.results.OTResult
        -------

        """

        if not (bool(self.geometries_inter_dict) or bool(self.geometries_intra_dict)):
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


