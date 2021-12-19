from typing import Any, Dict, List, Tuple, Union, Optional, Literal
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

from moscot._solver import FusedGW, RegularizedOT, Regularized
from moscot.framework.results import OTResult
from moscot.framework.BaseProblem import BaseProblem
from moscot.framework.utils import _verify_key, _prepare_geometry, _prepare_geometries, _check_arguments, _create_constant_weights


strategies_MatchingEstimator = Literal["pairwise"]

class OTEstimator(BaseProblem, RegularizedOT):
    def __init__(
        self,
        adata: AnnData,
        params: Dict,
        cost_fn: Optional[CostFn_t] = None,
        epsilon: Optional[Union[float, Epsilon]] = None,
        **kwargs: Any,
    ) -> None:

        super().__init__(adata=adata, params=params, cost_fn=cost_fn, epsilon=epsilon, **kwargs)

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
        pass


class MatchingEstimator(OTEstimator):
    def __init__(self,
                 adata: AnnData,
                 key: str,
                 params: Dict,
                 cost_fn: Optional[CostFn_t] = None,
                 epsilon: Optional[Union[float, Epsilon]] = None,
                 **kwargs: Any
                 ) -> None:
        self.key = key
        self.geometries = None
        self.a = None #TODO: check whether we can put them in class of higher order
        self.b = None
        self._solver = Regularized
        self._regularized = None
        super().__init__(adata=adata, params=params, cost_fn=cost_fn, epsilon=epsilon, **kwargs)

    def prepare(self,
                policy: Union[Tuple, List[Tuple], strategies_MatchingEstimator],
                rep: None,
                cost_fn: Union[CostFn, None],
                eps: Union[float, None],
                groups: Union[List[str], Tuple[str]],
                **kwargs: Any
                ) -> None:

        transport_sets = _verify_key(self.adata, self.key, policy)

        if isinstance(policy, Tuple) or policy == "pairwise":
            self.geometries = _prepare_geometry(self.adata, self.key, transport_sets, cost_fn, **kwargs)
        elif isinstance(policy, List):
            self.geometries = _prepare_geometries(self.adata, transport_sets, cost_fn, **kwargs)
        else:
            raise NotImplementedError

    def fit(self,
            a: Optional[Union[jnp.array, List[jnp.array]]] = None,
            b: Optional[Union[jnp.array, List[jnp.array]]] = None,
            **kwargs: Any,
            ):

        if self.geometries is None:
            raise ValueError("Please run 'fit()' first.")

        _check_arguments(a, b, self.geometries)

        if a is None:
            if isinstance(self.geometries, Geometry):
                self.a, self.b = _create_constant_weights(self.geometries)
            else:
                a, b = map(_create_constant_weights, self.geometries)
                self.a = list(a)
                self.b = list(b)

        if isinstance(self.geometries, Geometry):
            self._regularized = self._solver.fit(self.geometries, self.a, self.b)
        else:
            self._regularized = []
            for i in range(len(self.geometries)):
                self._regularized.append(self._solver.fit(self.geometries[i], self.a[i], self.b[i]))

        return OTResult(self.adata, self._regularized)




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

        _fused_gw = self._sovler.fit(
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
