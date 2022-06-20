from typing import Any, Tuple, Mapping, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from moscot.problems.base import BaseProblem, BaseCompoundProblem  # type: ignore[attr-defined]

import wrapt

__all__ = ["require_prepare", "require_solution", "wrap_prepare", "wrap_solve"]


# TODO(michalk8): refactor using stage
@wrapt.decorator
def require_solution(
    wrapped: Callable[[Any], Any], instance: "BaseProblem", args: Tuple[Any, ...], kwargs: Mapping[str, Any]
) -> Any:
    """Check whether problem has been solved."""
    from moscot.problems.base import OTProblem, BaseCompoundProblem  # type: ignore[attr-defined]

    if isinstance(instance, OTProblem) and instance.solution is None:
        raise RuntimeError("TODO: Run solve.")
    if isinstance(instance, BaseCompoundProblem) and instance.solutions is None:
        raise RuntimeError("TODO: Run solve.")
    return wrapped(*args, **kwargs)


@wrapt.decorator
def require_prepare(
    wrapped: Callable[[Any], Any], instance: "BaseCompoundProblem", args: Tuple[Any, ...], kwargs: Mapping[str, Any]
) -> Any:
    """Check whether problem has been prepared."""
    if instance._problems is None:
        raise RuntimeError("TODO: Run prepare.")
    return wrapped(*args, **kwargs)


@wrapt.decorator
def wrap_prepare(
    wrapped: Callable[[Any], Any], instance: "BaseProblem", args: Tuple[Any, ...], kwargs: Mapping[str, Any]
) -> Any:
    """Check and update the state when preparing :class:`moscot.problems.base.OTProblem`."""
    from moscot.problems.base._base_problem import ProblemKind, ProblemStage  # TODO: move ENUMs to this file

    _ = wrapped(*args, **kwargs)
    if instance._problem_kind == ProblemKind.UNKNOWN:
        raise RuntimeError("TODO: problem kind not set after prepare")
    instance._stage = ProblemStage.PREPARED
    return instance


@wrapt.decorator
def wrap_solve(
    wrapped: Callable[[Any], Any], instance: "BaseProblem", args: Tuple[Any, ...], kwargs: Mapping[str, Any]
) -> Any:
    """Check and update the state when solving :class:`moscot.problems.base.OTProblem`."""
    from moscot.problems.base._base_problem import ProblemStage

    print(instance._stage)
    if self._stage != ProblemStage.PREPARED or self._stage != ProblemStage.SOLVED:
        raise RuntimeError("TODO")
    _ = wrapped(*args, **kwargs)
    instance._stage = ProblemStage.SOLVED
    return instance
