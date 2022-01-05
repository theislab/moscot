from typing import Any, Dict, List, Tuple, Union, Literal, Optional
from numbers import Number

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
        scale: str = None,
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
            raise ValueError("The gene expression data in the AnnData object is not correctly saved in {}".format(self.rep))

        self.geometries_dict = _prepare_xy_geometries(self.adata, key=self.key, transport_sets=transport_sets, rep=self.rep, cost_fn=self.cost_fn, custom_cost_matrix_dict=custom_cost_matrix_dict, scale=scale, **kwargs)

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
        sinkhorn_kwargs: Optional[Union[List, Dict[Tuple, List]]] = {},
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
            self._solver_dict[tup].fit(self.geometries_dict[tup], self.a_dict[tup], self.b_dict[tup],
                                       tau_a=self.tau_a_dict[tup], tau_b=self.tau_b_dict[tup], **self.sinkorn_kwargs_dict[tup], **kwargs)

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
        trees: Dict[int, DiGraph],
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
        self.a_dict: Dict[Tuple, jnp.ndarray] = {}  # TODO: check whether we can put them in class of higher order
        self.b_dict: Dict[Tuple, jnp.ndarray] = {}
        self._solver_dict: Dict[Tuple, FusedGW] = {}
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
        custom_inter_cost_matrix_dict: Optional[Dict[Tuple, jnp.ndarray]] = None,
        custom_intra_cost_matrix_dict: Optional[Dict[Tuple, jnp.ndarray]] = None,
        scale = None,
        **kwargs: Any,
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
        self.key = key
        self._scale = kwargs.pop("scale", "max")
        if self.key not in self.adata.obs.columns:
            raise ValueError(f"The provided key {self.key} is not found in the AnnData object.")
        transport_sets = _verify_key(self._adata, self.key, policy)

        if not isinstance(getattr(self._adata, self.rep), np.ndarray):
            raise ValueError("The gene expression data in the AnnData object is not correctly saved in {}".format(self.rep))

        _check_arguments(a, b, self.geometries_dict)

        if a is None:  # TODO: discuss whether we want to have this here, i.e. whether we want to explicitly create weights because OTT would do this for us.
            # TODO: atm do it here to have all parameter saved in the estimator class
            self.a_dict = {tup: _create_constant_weights_source(geom) for tup, geom in self.geometries_dict.items()}
            self.b_dict = {tup: _create_constant_weights_target(geom) for tup, geom in self.geometries_dict.items()}


        self.tree_cost_dict = _prepare_xx_geometries(self.tree_dict, CostFn_tree, custom_intra_cost_matrix_dict, scale, **kwargs)
        self.rna_cost_dict = _prepare_xy_geometries(self.adata, self.key, transport_sets, self.rep, cost_fn=self.cost_fn,
                                                    custom_cost_matrix_dict=custom_inter_cost_matrix_dict, scale=scale, **kwargs)

        #TODO: add some tests here, e.g. costs should be positive
        return self

    def fit(
            self,
            epsilon: Optional[Union[List[Union[float, Epsilon]], float, Epsilon]] = 0.5,
            alpha: Optional[Union[List[float], float]] = 0.5,
            tau_a: Optional[Union[List, Dict[Tuple, List], float]] = 1.0,
            tau_b: Optional[Union[List, Dict[Tuple, List], float]] = 1.0,
            sinkhorn_kwargs: Optional[Union[List, Dict[Tuple, List]]] = None,
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

        if not (bool(self.tree_cost_dict) or bool(self.rna_cost_dict)):
            raise ValueError("Please run 'prepare()' first.")

        tuples = list(self.geometries_dict.keys())  # TODO: make a class attribute out of this
        self.epsilon_dict = get_param_dict(epsilon, tuples)
        self.tau_a_dict = get_param_dict(tau_a, tuples)
        self.tau_b_dict = get_param_dict(tau_b, tuples)
        self.alpha_dict = get_param_dict(alpha, tuples)
        self.sinkorn_kwargs_dict = get_param_dict(sinkhorn_kwargs, tuples)


        self._solver_dict = {tup: FusedGW(alpha=self.alpha_dict[tup], epsilon=self.epsilon_dict[tup]) for tup in tuples}
        for tup, geom in self.geometries_inter_dict.items():
            self._solver_dict[tup].fit(self.rna_cost_dict[tup], self.tree_cost_dict[tup[0]], self.tree_cost_dict[tup[1]],
                                       self.a_dict[tup], self.b_dict[tup], tau_a=self.tau_a_dict[tup], tau_b=self.tau_b_dict[tup], **self.sinkorn_kwargs_dict[tup], **kwargs)

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
        rep: str = "X",
        **kwargs: Any,
    ) -> None:

        super().__init__(adata=adata, rep=rep, **kwargs)


class SpatialMappingEstimator(OTEstimator):
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


