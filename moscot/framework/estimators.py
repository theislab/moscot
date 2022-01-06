from typing import Any, Dict, List, Tuple, Union, Literal, Optional
from numbers import Number
from abc import abstractmethod
from networkx import DiGraph
from jax import numpy as jnp
from ott.geometry.costs import CostFn
from ott.geometry.geometry import Geometry
from ott.core.gromov_wasserstein import GWLoss
import numpy as np
from ott.geometry.costs import CostFn, Euclidean

from ott.geometry.epsilon_scheduler import Epsilon

from anndata import AnnData

from moscot._solver import FusedGW, Regularized, RegularizedOT
from moscot.framework.utils import (
    _verify_key,
    _check_arguments,
    _prepare_xy_geometry,
    _prepare_xy_geometries,
    _create_constant_weights_source,
    _create_constant_weights_target,
    _prepare_geometries_from_cost,
    get_param_dict,
    _prepare_xx_geometries,
)
from moscot.framework.custom_costs import Leaf_distance
from moscot.framework.BaseProblem import BaseProblem
from moscot.framework.settings import strategies_MatchingEstimator
from moscot.framework.results import BaseResult, OTResult


CostFn_t = Union[CostFn, GWLoss]
CostFn_tree = Union[Leaf_distance]
CostFn_general = Union[CostFn_t, CostFn_tree]
Scales = Union["mean", "meadian", "max"]

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

        self.geometries_dict: Dict[Tuple, Geometry] = {}
        self.epsilon_dict: Dict[Tuple, Union[List[Union[float, Epsilon]], float, Epsilon]] = None
        self.sinkhorn_kwargs_dict: Dict[Tuple, Dict[str, Any]] = None
        self.a_dict: Dict[Tuple, jnp.ndarray] = {}
        self.b_dict: Dict[Tuple, jnp.ndarray] = {}
        self._key: str = None
        self.tau_a_dict: Optional[Dict[Tuple, jnp.ndarray]] = None
        self.tau_b_dict: Optional[Dict[Tuple, jnp.ndarray]] = None
        self.scale: Optional[Scales] = None
        self._kwargs: Dict[str, Any] = kwargs
        self._transport_sets: List[Tuple] = None

        super().__init__(adata=adata, rep=rep)

    def estimate_growth_rates(self) -> None:
        # https://github.com/broadinstitute/wot/blob/master/wot/gene_set_scores.py
        pass
    
    @property
    def transport_sets(self) -> List[Tuple]:
        return self._transport_sets
    
    @property
    def key(self) -> str:
        return self._key

    @property
    def solvers(self) -> Dict[Tuple, Any]:
        pass


class SinkhornEstimator(OTEstimator):
    """
    class handling all estimators using Sinkhorn / linear optimal transport
    """
    def __init__(
            self,
            adata: AnnData,
            rep: str = "X",
            **kwargs: Any,
    ) -> None:
        super().__init__(adata=adata, rep=rep)

        self._solver_dict: Dict[Tuple, Regularized] = {}
        self.cost_fn: CostFn_t = None

    @property
    def solvers(self) -> Dict[Tuple, Regularized]:  # we need it here because of return type ->Dict[Tuple, Regularized]
        return self._solver_dict

    @property
    def transport_matrix(self) -> Dict[Tuple, jnp.ndarray]:  # we need it here because self._solver_dict of type Dict[Tuple, Regularized] is initialized in this class
        return {tup: self._solver_dict[tup]._transport.matrix for tup in self._transport_sets} #TODO use getter functino for _transport


class GWEstimator(OTEstimator):
    """
    class handling all estimators using pure Gromov-Wasserstein
    """
    def __init__(
            self,
            adata: AnnData,
            rep: str = "X",
            **kwargs: Any,
    ) -> None:
        pass


class FGWEstimator(OTEstimator):
    """
    class handling all estimators using Fused Gromov Wasserstein
    """
    def __init__(
            self,
            adata: AnnData,
            rep: str = "X",
            **kwargs: Any,
    ) -> None:
        super().__init__(adata=adata, rep=rep)
        self._solver_dict: Dict[Tuple, FusedGW] = {}
        self.xy_cost_fn: CostFn_t = None
        self.xx_cost_fn: CostFn_general = None

    @property
    def solvers(self) -> Dict[Tuple, FusedGW]:
        return self._solver_dict

    @property
    def transport_matrix(self) -> Dict[Tuple, jnp.ndarray]:
        return {tup: self._solver_dict[tup]._transport.matrix for tup in self._transport_sets} #TODO: use getter fn for _transport


class MatchingEstimator(SinkhornEstimator):
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
        rep
            instance defining how the gene expression is saved in adata
        """

        super().__init__(adata=adata, rep=rep, **kwargs)

    def prepare(
        self,
        key: str,
        policy: Union[List[Tuple], strategies_MatchingEstimator],
        subset: List = None, # e.g. time points [1,3,5,7]
        a: Optional[Union[jnp.array, List[jnp.array]]] = None,
        b: Optional[Union[jnp.array, List[jnp.array]]] = None,
        cost_fn: Optional[CostFn_t] = Euclidean(),
        cost_matrix_dict: Optional[Dict[Tuple, jnp.ndarray]] = None,
        scale: str = None,
        **kwargs: Any,
    ) -> "MatchingEstimator":
        """

        Parameters
        ----------
        key
            column of AnnData.obs containing assignment of data points to distributions
        policy
            2-tuples of values of self._key defining the distribution which the optimal transport maps are calculated for
        subset
            If policy is not explicit, i.e. a list of tuples, but a strategy is given the strategy is applied to the
            subset of values given in the key column
        a:
            weights for source distribution. If of type jnp.ndarray the same distribution is taken for all models, if of type
            List[jnp.ndarray] the length of the list must be equal to the number of transport maps defined in prepare()
        b:
            weights for target distribution. If of type jnp.ndarray the same distribution is taken for all models, if of type
            List[jnp.ndarray] the length of the list must be equal to the number of transport maps defined in prepare()
        cost_fn
            cost function used to create the cost matrix for the OT problem
        cost_matrix_dict
            dictionary of custom cost matrices with keys corresponding to the transport tuple and value the corresponding
            cost matrix. If cost_matrix_dict is provided cost_fn is neglected
        scale
            how to scale the cost matrix, currently only provided for custom cost matrices
        **kwargs
            ott.geometry.Geometry kwargs

        Returns
            self
        -------
        """
        self._key = key
        self.cost_fn = cost_fn
        self._transport_sets = _verify_key(self._adata, self._key, policy, subset)
        self.scale = scale
        if not isinstance(getattr(self._adata, self.rep), np.ndarray):
            raise ValueError("The gene expression data in the AnnData object is not correctly saved in {}".format(self.rep))

        self.geometries_dict = _prepare_xy_geometries(self.adata,
                                                      key=self._key,
                                                      transport_sets=self._transport_sets,
                                                      rep=self.rep,
                                                      cost_fn=self.cost_fn,
                                                      custom_cost_matrix_dict=cost_matrix_dict,
                                                      scale=self.scale,
                                                      **kwargs)

        _check_arguments(a, b, self.geometries_dict)

        if a is None:
            self.a_dict = {tup: _create_constant_weights_source(geom) for tup, geom in self.geometries_dict.items()}
            self.b_dict = {tup: _create_constant_weights_target(geom) for tup, geom in self.geometries_dict.items()}

        return self

    def fit(
        self,
        epsilon: Optional[Union[List[Union[float, Epsilon]], float, Epsilon]] = 0.5,
        tau_a: Optional[Union[List, Dict[Tuple, List], float]] = 1.0,
        tau_b: Optional[Union[List, Dict[Tuple, List], float]] = 1.0,
        sinkhorn_kwargs: Optional[Union[List, Dict[Tuple, List]]] = {},
        **kwargs
    ) -> "OTResult":
        """

        Parameters
        ----------
        epsilon
            regularization parameter for OT problem
        tau_a
             ratio rho/(rho+eps) between KL divergence regularizer to first
             marginal and itself + epsilon regularizer used in the unbalanced
             formulation.
        tau_b:
             ratio rho/(rho+eps) between KL divergence regularizer to first
             marginal and itself + epsilon regularizer used in the unbalanced
             formulation.
        sinkhorn_kwargs
            estimator-specific kwargs for ott.core.sinkhorn.sinkhorn
        **kwargs
            ott.core.sinkhorn.sinkhorn keyword arguments applied to all estimators
        Returns
            moscot.framework.results.OTResult
        -------

        """

        if not bool(self.geometries_dict):
            raise ValueError("Please run 'prepare()' first.")

        self.epsilon_dict = get_param_dict(epsilon, self._transport_sets)
        self.tau_a_dict = get_param_dict(tau_a, self._transport_sets)
        self.tau_b_dict = get_param_dict(tau_b, self._transport_sets)
        self.sinkhorn_kwargs_dict = get_param_dict(sinkhorn_kwargs, self._transport_sets)

        self._solver_dict = {tup: Regularized(cost_fn=self.cost_fn, epsilon=self.epsilon_dict[tup]) for tup in self._transport_sets}
        for tup, geom in self.geometries_dict.items():
            self._solver_dict[tup].fit(self.geometries_dict[tup], self.a_dict[tup], self.b_dict[tup],
                                       tau_a=self.tau_a_dict[tup], tau_b=self.tau_b_dict[tup], **self.sinkhorn_kwargs_dict[tup], **kwargs)

        return OTResult(self.adata, self._key, self._solver_dict)


class LineageEstimator(FGWEstimator):
    """
    This estimator handles FGW estimators for temporal data
    """
    def __init__(
        self,
        adata: AnnData,
        tree_dict: Dict[Any, DiGraph],
        rep: str = "X",
        **kwargs: Any,
    ) -> None:
        """

        Parameters
        ----------
        adata
            AnnData object containing the gene expression for all data points
        tree_dict
            dictionary with keys being time points and values being the corresponding lineage tree
        rep
            instance defining how the gene expression is saved in the AnnData object

        Returns
            None
        """
        self.tree_dict = tree_dict
        super().__init__(adata=adata, rep=rep, **kwargs)

    def prepare(
        self,
        key: str,
        policy: Union[List[Tuple], strategies_MatchingEstimator],
        subset: List = None,  # e.g. time points [1,3,5,7]
        a: Optional[Union[jnp.array, List[jnp.array]]] = None,
        b: Optional[Union[jnp.array, List[jnp.array]]] = None,
        rna_cost_fn: Optional[CostFn_t] = Euclidean(),
        tree_cost_fn: Optional[CostFn_tree] = Leaf_distance(),
        xy_cost_matrix_dict: Optional[Dict[Tuple, jnp.ndarray]] = None,
        xx_cost_matrix_dict: Optional[Dict[Tuple, jnp.ndarray]] = None,
        scale = None,
        **kwargs: Any,
    ) -> None:
        """

        Parameters
        ----------
        policy
            2-tuples of values of self._key defining the distribution which the optimal transport maps are calculated for
        kwargs:
            kwargs for ott.geometry

        Returns
        -------

        """
        self._key = key
        self.scale = scale
        self.xy_cost_fn = rna_cost_fn
        self.xx_cost_fn = tree_cost_fn
        if self._key not in self.adata.obs.columns:
            raise ValueError(f"The provided key {self._key} is not found in the AnnData object.")
        self._transport_sets = _verify_key(self._adata, self._key, policy)

        if not isinstance(getattr(self._adata, self.rep), np.ndarray):
            raise ValueError("The gene expression data in the AnnData object is not correctly saved in {}".format(self.rep))

        self.xx_geometries_dict = _prepare_xx_geometries(self.tree_dict, self.xx_cost_fn, xx_cost_matrix_dict, self.scale, **kwargs)
        self.xy_geometries_dict = _prepare_xy_geometries(self.adata, self._key, self._transport_sets, self.rep, cost_fn=self.xy_cost_fn,
                                                    custom_cost_matrix_dict=xy_cost_matrix_dict, scale=self.scale, **kwargs)


        _check_arguments(a, b, self.xy_geometries_dict)

        if a is None:  # TODO: discuss whether we want to have this here, i.e. whether we want to explicitly create weights because OTT would do this for us.
            # TODO: atm do it here to have all parameter saved in the estimator class
            self.a_dict = {tup: _create_constant_weights_source(geom) for tup, geom in self.xy_geometries_dict.items()}
            self.b_dict = {tup: _create_constant_weights_target(geom) for tup, geom in self.xy_geometries_dict.items()}

        #TODO: add some tests here, e.g. costs should be positive
        return self

    def fit(
            self,
            epsilon: Optional[Union[List[Union[float, Epsilon]], float, Epsilon]] = 0.5,
            alpha: Optional[Union[List[float], float]] = 0.5,
            tau_a: Optional[Union[List, Dict[Tuple, List], float]] = 1.0,
            tau_b: Optional[Union[List, Dict[Tuple, List], float]] = 1.0,
            sinkhorn_kwargs: Optional[Union[List, Dict[Tuple, List]]] = {},
            **kwargs: Any,
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
        if tau_a != 1 or tau_b != 1:
            raise NotImplementedError("Currently, only balanced problems are supported for GW and FGW problems.")

        if not (bool(self.xx_geometries_dict) or bool(self.xy_geometries_dict)):
            raise ValueError("Please run 'prepare()' first.")

        self.epsilon_dict = get_param_dict(epsilon, self._transport_sets)
        self.tau_a_dict = get_param_dict(tau_a, self._transport_sets)
        self.tau_b_dict = get_param_dict(tau_b, self._transport_sets)
        self.alpha_dict = get_param_dict(alpha, self._transport_sets)
        self.sinkhorn_kwargs_dict = get_param_dict(sinkhorn_kwargs, self._transport_sets)

        self._solver_dict = {tup: FusedGW(alpha=self.alpha_dict[tup], epsilon=self.epsilon_dict[tup]) for tup in self._transport_sets}
        for tup, geom in self.xy_geometries_dict.items():
            self._solver_dict[tup].fit(self.xx_geometries_dict[tup[0]], self.xx_geometries_dict[tup[1]], self.xy_geometries_dict[tup],
                                       self.a_dict[tup], self.b_dict[tup], tau_a=self.tau_a_dict[tup], tau_b=self.tau_b_dict[tup], **self.sinkhorn_kwargs_dict[tup], **kwargs)

        return OTResult(self.adata, self._key, self._solver_dict)


class SpatialAlignmentEstimator(FGWEstimator):
    """
    This estimator ...
    """
    def __init__(
        self,
        adata: AnnData,
        rep: str = "X",
        **kwargs: Any,
    ) -> None:

        super().__init__(adata=adata, rep=rep, **kwargs)


class SpatialMappingEstimator(FGWEstimator):
    """
    This estimator ...
    """
    def __init__(
        self,
        adata: AnnData,
        rep: str = "X",
        **kwargs: Any,
    ) -> None:

        super().__init__(adata=adata, rep=rep, **kwargs)


