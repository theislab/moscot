import types
from types import MappingProxyType
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Tuple, Type, Union

from anndata import AnnData

from moscot import _constants
from moscot._types import (
    CostKwargs_t,
    OttCostFn_t,
    OttCostFnMap_t,
    Policy_t,
    ProblemStage_t,
    QuadInitializer_t,
    ScaleCost_t,
    SinkhornInitializer_t,
)
from moscot.base.problems.compound_problem import B, Callback_t, CompoundProblem, K
from moscot.base.problems.problem import CondOTProblem, OTProblem
from moscot.problems._utils import (
    handle_conditional_attr,
    handle_cost,
    handle_cost_tmp,
    handle_joint_attr,
    handle_joint_attr_tmp,
)
from moscot.problems.generic._mixins import GenericAnalysisMixin

__all__ = ["GENOTLinProblem"]


class GENOTLinProblem(CondOTProblem):
    """Class for solving Conditional Parameterized Monge Map problems / Conditional Neural OT problems."""

    def prepare(
        self,
        key: str,
        joint_attr: Union[str, Mapping[str, Any]],
        conditional_attr: Union[str, Mapping[str, Any]],
        policy: Literal["sequential", "star", "explicit"] = "sequential",
        a: Optional[str] = None,
        b: Optional[str] = None,
        cost: OttCostFn_t = "sq_euclidean",
        cost_kwargs: CostKwargs_t = types.MappingProxyType({}),
        **kwargs: Any,
    ) -> "GENOTLinProblem":
        """Prepare the :class:`moscot.problems.generic.GENOTLinProblem`."""
        self.batch_key = key
        xy, kwargs = handle_joint_attr_tmp(joint_attr, kwargs)
        conditions = handle_conditional_attr(conditional_attr)
        xy, xx = handle_cost_tmp(xy=xy, x={}, y={}, cost=cost, cost_kwargs=cost_kwargs)
        return super().prepare(
            policy_key=key,
            policy=policy,
            xy=xy,
            xx=xx,
            conditions=conditions,
            a=a,
            b=b,
            **kwargs,
        )

    def solve(
        self,
        batch_size: int = 1024,
        seed: int = 0,
        iterations: int = 25000,  # TODO(@MUCDK): rename to max_iterations
        valid_freq: int = 50,
        valid_sinkhorn_kwargs: Dict[str, Any] = MappingProxyType({}),
        train_size: float = 1.0,
        **kwargs: Any,
    ) -> "GENOTLinProblem":
        """Solve."""
        return super().solve(
            batch_size=batch_size,
            # tau_a=tau_a, # TODO: unbalancedness handler
            # tau_b=tau_b,
            seed=seed,
            n_iters=iterations,
            valid_freq=valid_freq,
            valid_sinkhorn_kwargs=valid_sinkhorn_kwargs,
            train_size=train_size,
            solver_name="GENOTLinSolver",
            **kwargs,
        )

    @property
    def _base_problem_type(self) -> Type[CondOTProblem]:
        return CondOTProblem

    @property
    def _valid_policies(self) -> Tuple[Policy_t, ...]:
        return _constants.SEQUENTIAL, _constants.EXPLICIT  # type: ignore[return-value]
