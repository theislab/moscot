from typing import Any, List, Type, Tuple, Union, Literal, Mapping, Optional

from moscot._types import Numeric_t
from moscot._docs._docs import d
from moscot.problems._utils import handle_joint_attr
from moscot._constants._constants import Policy
from moscot.problems.base._birth_death import BirthDeathMixin, BirthDeathProblem, BirthDeathNeuralProblem
from moscot.problems.base._compound_problem import CompoundProblem


@d.dedent
class TemporalNeuralProblem(BirthDeathMixin, CompoundProblem[Numeric_t, BirthDeathProblem]):
    """TemporalNeuralProblem."""

    @d.dedent
    def prepare(
        self,
        time_key: str,
        joint_attr: Union[str, Mapping[str, Any]],
        policy: Literal["sequential", "tril", "triu", "explicit"] = "sequential",
        **kwargs: Any,
    ) -> "TemporalNeuralProblem":
        """Prepare the :class:`moscot.problems.time.TemporalNeuralProblem`."""
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
        seed: int = 0,
        pos_weights: bool = False,
        dim_hidden: Optional[List[int]] = None,
        beta: float = 1.0,
        metric: str = "sinkhorn_forward",
        learning_rate: float = 1e-3,
        beta_one: float = 0.5,
        beta_two: float = 0.9,
        weight_decay: float = 0.0,
        iterations: int = 25000,
        inner_iters: int = 10,
        valid_freq: int = 50,
        log_freq: int = 5,
        patience: int = 100,
        pretrain: bool = True,
        pretrain_iters: int = 15001,
        pretrain_scale: float = 3.0,
        train_size: float = 1.0,
        batch_size: int = 1024,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        **kwargs: Any,
    ) -> "TemporalNeuralProblem":
        """Solve."""
        return super().solve(
            seed=seed,
            pos_weights=pos_weights,
            beta=beta,
            pretrain=pretrain,
            metric=metric,
            iterations=iterations,
            inner_iters=inner_iters,
            valid_freq=valid_freq,
            log_freq=log_freq,
            patience=patience,
            dim_hidden=dim_hidden,
            learning_rate=learning_rate,
            beta_one=beta_one,
            beta_two=beta_two,
            weight_decay=weight_decay,
            pretrain_iters=pretrain_iters,
            pretrain_scale=pretrain_scale,
            train_size=train_size,
            batch_size=batch_size,
            tau_a=tau_a,
            tau_b=tau_b,
            **kwargs,
        )  # type:ignore[return-value]

    @property
    def _base_problem_type(self) -> Type[BirthDeathProblem]:
        return BirthDeathNeuralProblem

    @property
    def _valid_policies(self) -> Tuple[str, ...]:
        return Policy.SEQUENTIAL, Policy.TRIU, Policy.EXPLICIT
