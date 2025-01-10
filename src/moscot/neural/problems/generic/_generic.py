import types
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    Hashable,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from moscot import _constants
from moscot._types import CostKwargs_t, OttCostFn_t, Policy_t
from moscot.neural.base.problems.problem import NeuralOTProblem

__all__ = ["GENOTLinProblem"]

K = TypeVar("K", bound=Hashable)


class GENOTLinProblem(NeuralOTProblem[K]):
    """Class for solving Conditional Parameterized Monge Map problems / Conditional Neural OT problems."""

    def prepare(
        self,
        key: str,
        joint_attr: Union[str, Mapping[str, Any]],
        condition_attr: Union[str, Mapping[str, Any]] = None,
        src_quad_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        tgt_quad_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        policy: Literal["sequential", "star", "explicit"] = "sequential",
        cost: OttCostFn_t = "sq_euclidean",
        cost_kwargs: CostKwargs_t = types.MappingProxyType({}),
        **kwargs: Any,
    ) -> "GENOTLinProblem[K]":
        """Prepare the :class:`moscot.problems.generic.GENOTLinProblem`."""
        self.batch_key = key
        # TODO: These cost functions should be going to GENOT match_function somehow
        del cost, cost_kwargs
        lin = GENOTLinProblem._handle_attr(joint_attr)
        src_quad = GENOTLinProblem._handle_attr(src_quad_attr) if src_quad_attr is not None else None
        tgt_quad = GENOTLinProblem._handle_attr(tgt_quad_attr) if tgt_quad_attr is not None else None
        condition = GENOTLinProblem._handle_attr(condition_attr) if condition_attr is not None else None
        return super().prepare(
            policy_key=key,
            policy=policy,
            lin=lin,
            src_quad=src_quad,
            tgt_quad=tgt_quad,
            condition=condition,
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
    ) -> "GENOTLinProblem[K]":
        """Solve."""
        return super().solve(
            batch_size=batch_size,
            seed=seed,
            n_iters=iterations,
            valid_freq=valid_freq,
            valid_sinkhorn_kwargs=valid_sinkhorn_kwargs,
            train_size=train_size,
            solver_name="GENOTSolver",
            **kwargs,
        )

    @property
    def _base_problem_type(self) -> Type[NeuralOTProblem[K]]:
        return NeuralOTProblem[K]

    @property
    def _valid_policies(self) -> Tuple[Policy_t, ...]:
        return _constants.SEQUENTIAL, _constants.EXPLICIT  # type: ignore[return-value]
