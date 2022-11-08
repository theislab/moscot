from typing import Any, Type, Tuple, Union, Literal, Mapping

from anndata import AnnData

from moscot._types import Numeric_t
from moscot._docs._docs import d
from moscot.problems._utils import handle_joint_attr
from moscot._constants._constants import Policy
from moscot.problems.base._base_problem import NeuralOTProblem
from moscot.problems.base._compound_problem import B, CompoundProblem


@d.dedent
class TemporalNeuralProblem(CompoundProblem[Numeric_t, B]):
    """TemporalNeuralProblem."""

    def __init__(self, adata: AnnData, **kwargs: Any):
        super().__init__(adata, **kwargs)

    @d.dedent
    def prepare(
        self,
        time_key: str,
        joint_attr: Union[str, Mapping[str, Any]],
        policy: Literal["sequential", "tril", "triu", "explicit"] = "sequential",
        **kwargs: Any,
    ) -> "TemporalNeuralProblem[Any]":
        """Prepare the :class:`moscot.problems.time.TemporalProblem`."""
        self.temporal_key = time_key
        policy = Policy(policy)  # type: ignore[assignment]
        xy, kwargs = handle_joint_attr(joint_attr, kwargs)

        return super().prepare(
            key=time_key,
            xy=xy,
            policy=policy,
            **kwargs,
        )

    @d.dedent
    def solve(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> "TemporalNeuralProblem[Any]":
        """Solve."""
        return super().solve(
            *args,
            **kwargs,
        )  # type:ignore[return-value]

    @property
    def _base_problem_type(self) -> Type[B]:
        return NeuralOTProblem  # type: ignore[return-value]

    @property
    def _valid_policies(self) -> Tuple[str, ...]:
        return Policy.SEQUENTIAL, Policy.TRIU, Policy.EXPLICIT
