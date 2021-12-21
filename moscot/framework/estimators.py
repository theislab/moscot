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
    _create_constant_weights_target
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
        self._regularized_dict = None
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

        self.geometries_dict = _prepare_geometries(self.adata, self.key, transport_sets, cost_fn, **kwargs)

    def fit(
        self,
        a: Optional[Union[jnp.array, List[jnp.array]]] = None,
        b: Optional[Union[jnp.array, List[jnp.array]]] = None,
        **kwargs: Any,
    ) -> "OTResult":

        if self.geometries is None:
            raise ValueError("Please run 'prepare()' first.")

        _check_arguments(a, b, self.geometries_dict)

        if a is None: #TODO: discuss whether we want to have this here, i.e. whether we want to explicitly create weights because OTT would do this for us.
            #TODO: atm do it here to have all parameter saved in the estimator class
            self.a_dict = {tup: _create_constant_weights_source(geom) for tup, geom in self.geometries_dict.items()}
            self.b_dict = {tup: _create_constant_weights_target(geom) for tup, geom in self.geometries_dict.items()}

        #if isinstance(self.geometries, Geometry):
        #    self._solver = Regularized(cost_fn=self.cost_fn, epsilon=self.epsilon)
        #    self._regularized = self._solver.fit(self.geometries, a=self.a, b=self.b)
        #else:
        self._solver_dict = {tup: Regularized(cost_fn=self.cost_fn, epsilon=self.epsilon) for tup, _ in self.geometries_dict.items()}
        self._regularized_dict = {}
        for tup, geom in self.geometries_dict.items():
            self._regularized_dict[tup] = self._solver_dict[tup].fit(self.geometries_dict[tup], self.a_dict[tup], self.b_dict[tup])

        return OTResult(self.adata, self._regularized_dict)

    def converged(self) -> Optional[bool]:
        pass

    def matrix(self) -> jnp.ndarray:
        pass

    def transport(self, inputs: jnp.ndarray, forward: bool = True) -> jnp.ndarray:
        pass


class LineageEstimator(OTEstimator):
    def __init__(
        self,
        adata: AnnData,
        params: Dict,
        cost_fn: Optional[CostFn_t] = None,
        epsilon: Optional[Union[float, Epsilon]] = None,
        alpha: Number = None,
        tree_rep: Union[str, DiGraph, None] = None,
        tree_cost: Union["MLE", "edgesum", "uniform_edge"] = None,
        **kwargs: Any,
    ) -> None:
        self.alpha = alpha
        self.tree_rep = tree_rep
        self.tree_cost = tree_cost
        self._solver = FusedGW(alpha=alpha)
        super().__init__(adata=adata, params=params, cost_fn=cost_fn, epsilon=epsilon, **kwargs)

    def prepare(
        self,
        key: Union[str, None],
        policy: None,
        rep: None,
        cost_fn: Union[CostFn, None],
        eps: Union[float, None],
        groups: Union[List[str], Tuple[str]],
    ) -> None:
        if key not in self.adata.obs.columns:
            raise ValueError(f"The provided key {key} is not found in the AnnData object.")

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
        self.alpha = alpha
        self.spatial_rep = spatial_rep
        self.spatial_cost = spatial_cost
        super().__init__(adata=adata, params=params, cost_fn=cost_fn, epsilon=epsilon, **kwargs)


class SpatialMappingEstimator(OTEstimator):
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
        self.alpha = alpha
        self.spatial_rep = spatial_rep
        self.spatial_cost = spatial_cost
        self.reference_var = reference_var
        super().__init__(adata=adata, params=params, cost_fn=cost_fn, epsilon=epsilon, **kwargs)


