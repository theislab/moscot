from typing import Any, Dict, List, Tuple, Union, Optional

from jax import numpy as jnp
from ott.geometry.costs import CostFn, Euclidean
from ott.geometry.geometry import Geometry
from ott.core.gromov_wasserstein import GWLoss
from ott.geometry.epsilon_scheduler import Epsilon
import numpy as np

from anndata import AnnData

from moscot._solver import Regularized
from old_code_to_recycle.framework.settings import strategies_MatchingEstimator
from old_code_to_recycle.framework.geom.utils import _prepare_xy_geometries
from old_code_to_recycle.framework.utils.utils import (
    _verify_key,
    get_param_dict,
    _check_arguments,
    _create_constant_weights_source,
    _create_constant_weights_target,
)
from old_code_to_recycle.framework.results._results import OTResult
from old_code_to_recycle.framework.utils.custom_costs import Leaf_distance
from old_code_to_recycle.framework.problems.BaseProblem import BaseProblem
from old_code_to_recycle.framework.results._result_mixins import ResultMixin

CostFn_t = Union[CostFn, GWLoss]
CostFn_tree = Union[Leaf_distance]
CostFn_general = Union[CostFn_t, CostFn_tree]
Scales = Union["mean", "median", "max"]


class SimpleProblem(BaseProblem, ResultMixin):
    """
    This estimator handles linear OT problems
    """

    def __init__(
        self,
        adata: AnnData,
        **kwargs: Any,
    ) -> None:
        """

        Parameters
        ----------
        adata
            AnnData object containing the gene expression for all data points
        """
        self._solver_dict: Dict[Tuple, Regularized] = {}
        self.cost_fn: CostFn_t = None
        self.geometries_dict: Dict[Tuple, Geometry] = {}
        self.epsilon_dict: Dict[Tuple, Union[List[Union[float, Epsilon]], float, Epsilon]] = None
        self.sinkhorn_kwargs_dict: Dict[Tuple, Dict[str, Any]] = None
        self.a_dict: Dict[Tuple, np.ndarray] = {}
        self.b_dict: Dict[Tuple, np.ndarray] = {}
        self._key: str = None
        self.tau_a_dict: Optional[Dict[Tuple, np.ndarray]] = None
        self.tau_b_dict: Optional[Dict[Tuple, np.ndarray]] = None
        self.scale: Optional[Scales] = None
        self._kwargs: Dict[str, Any] = kwargs
        self._transport_sets: List[Tuple] = None
        super().__init__(adata=adata, **kwargs)

    @property
    def solvers(
        self,
    ) -> Dict[Tuple, Regularized]:  # we need it here because of return type ->Dict[Tuple, **Regularized**]
        return self._solver_dict

    @property
    def transport_matrix(
        self,
    ) -> Dict[
        Tuple, np.ndarray
    ]:  # we need it here because self._solver_dict of type Dict[Tuple, Regularized] is initialized in this class
        return {
            tup: self._solver_dict[tup]._transport.matrix for tup in self._transport_sets
        }  # TODO use getter function for _transport

    def prepare(
        self,
        key: str,
        policy: Union[List[Tuple], strategies_MatchingEstimator],
        subset: List = None,  # e.g. time points [1,3,5,7]
        a: Optional[Union[jnp.array, List[jnp.array]]] = None,
        b: Optional[Union[jnp.array, List[jnp.array]]] = None,
        cost_fn: Optional[CostFn_t] = Euclidean(),
        cost_matrix_dict: Optional[Dict[Tuple, np.ndarray]] = None,
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
            weights for source distribution. If of type np.ndarray the same distribution is taken for all models, if of type
            List[np.ndarray] the length of the list must be equal to the number of transport maps defined in prepare()
        b:
            weights for target distribution. If of type np.ndarray the same distribution is taken for all models, if of type
            List[np.ndarray] the length of the list must be equal to the number of transport maps defined in prepare()
        cost_fn
            cost function used to create the cost matrix for the OT problem
        cost_matrix_dict
            dictionary of custom cost matrices with keys corresponding to the transport tuple and value the corresponding
            cost matrix. If cost_matrix_dict is provided cost_fn is neglected
        scale
            how to scale the cost matrix, currently only provided for custom cost matrices
        **kwargs
            moscot.framework.geom.geometry.geom kwargs

        Returns
            self
        -------
        """
        self._key = key
        self.cost_fn = cost_fn
        self._transport_sets = _verify_key(self._adata, self._key, policy, subset)
        self.scale = scale
        if not isinstance(getattr(self._adata, self.rep), np.ndarray):
            raise ValueError(f"The gene expression data in the AnnData object is not correctly saved in {self.rep}")

        self.geometries_dict = _prepare_xy_geometries(
            self.adata,
            key=self._key,
            transport_sets=self._transport_sets,
            rep=self.rep,
            cost_fn=self.cost_fn,
            custom_cost_matrix_dict=cost_matrix_dict,
            scale=self.scale,
            **kwargs,
        )

        _check_arguments(a, b, self.geometries_dict)

        if a is None:
            self.a_dict = {tup: _create_constant_weights_source(geom) for tup, geom in self.geometries_dict.items()}
            self.b_dict = {tup: _create_constant_weights_target(geom) for tup, geom in self.geometries_dict.items()}

        return self

    def solve(
        self,
        epsilon: Optional[Union[List[Union[float, Epsilon]], float, Epsilon]] = 0.5,
        tau_a: Optional[Union[List, Dict[Tuple, List], float]] = 1.0,
        tau_b: Optional[Union[List, Dict[Tuple, List], float]] = 1.0,
        sinkhorn_kwargs: Optional[Union[List, Dict[Tuple, List]]] = {},
        **kwargs,
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

        self._solver_dict = {
            tup: Regularized(cost_fn=self.cost_fn, epsilon=self.epsilon_dict[tup]) for tup in self._transport_sets
        }
        for tup, geom in self.geometries_dict.items():
            self._solver_dict[tup].fit(
                self.geometries_dict[tup],
                self.a_dict[tup],
                self.b_dict[tup],
                tau_a=self.tau_a_dict[tup],
                tau_b=self.tau_b_dict[tup],
                **self.sinkhorn_kwargs_dict[tup],
                **kwargs,
            )

        return OTResult(self.adata, self._key, self._solver_dict)
