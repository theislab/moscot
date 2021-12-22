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
from moscot.framework.BaseProblem import BaseProblem
from moscot.framework.settings import strategies_MatchingEstimator
from moscot.framework.results import BaseResult, OTResult


class OTEstimator(BaseProblem, RegularizedOT):
    def __init__(
        self,
        adata: AnnData,
        params: Dict,
        cost_fn: Optional[CostFn_t] = None,
        epsilon: Optional[Union[float, Epsilon]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(adata=adata, cost_fn=cost_fn, epsilon=epsilon, params=params)
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
        self.key = key
        self.geometries_dict = None
        self.a_dict = None  # TODO: check whether we can put them in class of higher order
        self.b_dict = None
        self._solver_dict = None
        self.rep = None
        super().__init__(adata=adata, cost_fn=cost_fn, epsilon=epsilon, params=params, **kwargs)

    def prepare(
        self,
        policy: Union[Tuple, List[Tuple], strategies_MatchingEstimator],
        rep: Union[str, None] = "X",
        cost_fn: Union[CostFn, None] = None,
        eps: Union[float, None] = None,
        groups: Union[List[str], Tuple[str]] = None,
        **kwargs: Any,
    ) -> None:

        transport_sets = _verify_key(self._adata, self.key, policy)
        if not isinstance(getattr(self._adata, rep), np.ndarray):
            raise ValueError("Please provide a valid layer from the")

        self.geometries_dict = _prepare_geometries(self.adata, self.key, transport_sets, self.rep, cost_fn, **kwargs)

    def fit(
        self,
        a: Optional[Union[jnp.array, List[jnp.array]]] = None,
        b: Optional[Union[jnp.array, List[jnp.array]]] = None,
        **kwargs: Any,
    ) -> "OTResult":

        if self.geometries_dict is None:
            raise ValueError("Please run 'prepare()' first.")

        _check_arguments(a, b, self.geometries_dict)

        if a is None: #TODO: discuss whether we want to have this here, i.e. whether we want to explicitly create weights because OTT would do this for us.
            #TODO: atm do it here to have all parameter saved in the estimator class
            self.a_dict = {tup: _create_constant_weights_source(geom) for tup, geom in self.geometries_dict.items()}
            self.b_dict = {tup: _create_constant_weights_target(geom) for tup, geom in self.geometries_dict.items()}

        self._solver_dict = {tup: Regularized(cost_fn=self.cost_fn, epsilon=self.epsilon) for tup, _ in self.geometries_dict.items()}
        for tup, geom in self.geometries_dict.items():
            self._solver_dict[tup] = self._solver_dict[tup].fit(self.geometries_dict[tup], self.a_dict[tup], self.b_dict[tup])

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
        params: Dict,
        cost_fn: Optional[CostFn_t] = None,
        epsilon: Optional[Union[float, Epsilon]] = None,
        alpha: Number = 0.5, # TODO: adapt from paper
        tree_rep: Union[str, DiGraph, None] = None,
        tree_cost: Union["MLE", "edgesum", "uniform_edge"] = None,
        **kwargs: Any,
    ) -> None:
        self.key = key
        self.alpha = alpha
        self.tree_rep = tree_rep
        self.tree_cost = tree_cost
        self.geometries_xy_dict: Dict[Tuple, Geometry] = None
        self.geometries_xx_dict: Dict[Tuple, Geometry] = None
        self.geometries_yy_dict: Dict[Tuple, Geometry] = None
        self.a_dict: Dict[Tuple, jnp.ndarray] = None  # TODO: check whether we can put them in class of higher order
        self.b_dict: Dict[Tuple, jnp.ndarray] = None
        self.cost_xx_dict: Dict[Tuple, jnp.ndarray] = None
        self.cost_yy_dict: Dict[Tuple, jnp.ndarray] = None
        self._solver_dict: Dict[Tuple, FusedGW] = None
        super().__init__(adata=adata, cost_fn=cost_fn, epsilon=epsilon, params=params, **kwargs)


    def prepare(
        self,
        key: str,
        policy: Union[Tuple, List[Tuple]],
        rep: str, # representation of trees
        trees: List,
        cost_fn: Union[CostFn, None], # cost function for linear problem
        eps: Union[float, None],
        groups: Union[List[str], Tuple[str]],
        **kwargs
    ) -> None:
        self._scale = kwargs.pop("scale", "max")
        if key not in self.adata.obs.columns:
            raise ValueError(f"The provided key {key} is not found in the AnnData object.")
        transport_sets = _verify_key(self._adata, self.key, policy)
        #if not isinstance(getattr(self._adata, rep), np.ndarray):
        #    raise ValueError("Please provide a valid layer from the")

        self.geometries_xy_dict = _prepare_geometries(self.adata, self.key, transport_sets, self.rep, cost_fn, **kwargs)
        self.geometries_xx_dict = _prepare_geometries_from_cost(self.cost_xx_dict,
                                                                scale=self._scale)  # TODO: here we assume we can never save it as online=True
        self.geometries_yy_dict = _prepare_geometries_from_cost(self.cost_yy_dict, scale=self._scale)

    def fit(
        self,
        geom: Union[jnp.array, Geometry],
        a: Optional[jnp.array] = None,
        b: Optional[jnp.array] = None,
        **kwargs: Any,
    ) -> BaseResult:

        _fused_gw = self._solver.fit(
            geom_a=geom_a,
            geom_b=geom_b,
            geom_ab=geom_ab,
            a=a,
            b=b,
            max_iterations=max_iterations,
            rtol=rtol,
            atol=atol,
            init_method=init_method,
            scale_fn=scale_fn,
            linesearch=linesearch,
            seed=seed,
            verbose=verbose,
            **kwargs,
        )


class SpatialAlignmentEstimator(OTEstimator):
    """
    This estimator ...
    """
    def __init__(
        self,
        adata: AnnData,
        params: Dict,
        cost_fn: Optional[CostFn_t] = None,
        epsilon: Optional[Union[float, Epsilon]] = None,
        alpha: Number = None,
        spatial_rep: Union[str, np.ndarray, None] = None,
        spatial_cost: Union["spatial_cost", np.ndarray, None] = None,  # TODO: specify spatial cost
        **kwargs: Any,
    ) -> None:

        super().__init__(adata=adata, cost_fn=cost_fn, epsilon=epsilon, params=params, **kwargs)


class SpatialMappingEstimator(OTEstimator):
    """
    This estimator ...
    """
    def __init__(
        self,
        adata: AnnData,
        params: Dict,
        cost_fn: Optional[CostFn_t] = None,
        epsilon: Optional[Union[float, Epsilon]] = None,
        alpha: Number = None,
        spatial_rep: Union[str, np.ndarray, None] = None,
        spatial_cost: Union["spatial_cost", np.ndarray, None] = None,  # TODO: specify spatial cost
        reference_var: Union[List, None] = None,
        **kwargs: Any,
    ) -> None:

        super().__init__(adata=adata, cost_fn=cost_fn, epsilon=epsilon, params=params, **kwargs)


