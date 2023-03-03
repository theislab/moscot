from types import MappingProxyType
from typing import Any, Dict, Type, Tuple, Union, Literal, Mapping, Iterable, Optional

from moscot._types import Numeric_t
from moscot._docs._docs import d
from moscot.problems._utils import handle_joint_attr
from moscot._constants._constants import Policy
from moscot.problems.time._mixins import NeuralAnalysisMixin
from moscot.problems.base._birth_death import BirthDeathMixin, BirthDeathNeuralProblem
from moscot.problems.base._compound_problem import CompoundProblem


@d.dedent
class TemporalNeuralProblem(
    NeuralAnalysisMixin[Numeric_t, BirthDeathNeuralProblem],
    BirthDeathMixin,
    CompoundProblem[Numeric_t, BirthDeathNeuralProblem],
):
    """TemporalNeuralProblem."""

    @d.dedent
    def prepare(
        self,
        time_key: str,
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        policy: Literal["sequential", "tril", "triu", "explicit"] = "sequential",
        a: Optional[str] = None,
        b: Optional[str] = None,
        marginal_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> "TemporalNeuralProblem":
        """Prepare the :class:`moscot.problems.time.TemporalNeuralProblem`."""
        self.temporal_key = time_key
        policy = Policy(policy)  # type: ignore[assignment]
        xy, kwargs = handle_joint_attr(joint_attr, kwargs)

        # TODO(michalk8): needs to be modified
        marginal_kwargs = dict(kwargs.pop("marginal_kwargs", {}))
        marginal_kwargs["proliferation_key"] = self.proliferation_key
        marginal_kwargs["apoptosis_key"] = self.apoptosis_key
        if a is None:
            a = self.proliferation_key is not None or self.apoptosis_key is not None
        if b is None:
            b = self.proliferation_key is not None or self.apoptosis_key is not None

        return super().prepare(
            key=time_key,
            xy=xy,
            policy=policy,
            a=a,
            b=b,
            marginal_kwargs=marginal_kwargs,
            **kwargs,
        )

    @d.dedent
    def solve(
        self,
        batch_size: int = 1024,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        epsilon: float = 0.1,
        seed: int = 0,
        pos_weights: bool = False,
        dim_hidden: Iterable[int] = (64, 64, 64, 64),
        beta: float = 1.0,
        best_model_metric: Literal[
            "sinkhorn_forward", "sinkhorn"
        ] = "sinkhorn_forward",  # TODO(@MUCDK) include only backward sinkhorn
        iterations: int = 25000,  # TODO(@MUCDK): rename to max_iterations
        inner_iters: int = 10,
        valid_freq: int = 50,
        log_freq: int = 5,
        patience: int = 100,
        optimizer_f_kwargs: Dict[str, Any] = MappingProxyType({}),
        optimizer_g_kwargs: Dict[str, Any] = MappingProxyType({}),
        pretrain_iters: int = 15001,
        pretrain_scale: float = 3.0,
        combiner_kwargs: Dict[str, Any] = MappingProxyType({}),
        valid_sinkhorn_kwargs: Dict[str, Any] = MappingProxyType({}),
        train_size: float = 1.0,
        **kwargs: Any,
    ) -> "TemporalNeuralProblem":
        """Solve."""
        return super().solve(
            batch_size=batch_size,
            tau_a=tau_a,
            tau_b=tau_b,
            epsilon=epsilon,
            seed=seed,
            pos_weights=pos_weights,
            dim_hidden=dim_hidden,
            beta=beta,
            best_model_metric=best_model_metric,
            iterations=iterations,
            inner_iters=inner_iters,
            valid_freq=valid_freq,
            log_freq=log_freq,
            patience=patience,
            optimizer_f_kwargs=optimizer_f_kwargs,
            optimizer_g_kwargs=optimizer_g_kwargs,
            pretrain_iters=pretrain_iters,
            pretrain_scale=pretrain_scale,
            combiner_kwargs=combiner_kwargs,
            valid_sinkhorn_kwargs=valid_sinkhorn_kwargs,
            train_size=train_size,
            **kwargs,
        )  # type:ignore[return-value]

    @property
    def _base_problem_type(self) -> Type[BirthDeathNeuralProblem]:
        return BirthDeathNeuralProblem

    @property
    def _valid_policies(self) -> Tuple[str, ...]:
        return Policy.SEQUENTIAL, Policy.TRIU, Policy.EXPLICIT
