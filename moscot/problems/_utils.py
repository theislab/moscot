from typing import Any, Tuple, Mapping, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from moscot.problems.base import BaseProblem, BaseCompoundProblem  # type: ignore[attr-defined]

import wrapt

__all__ = ["require_prepare", "require_solution"]


@wrapt.decorator
def require_solution(
    wrapped: Callable[[Any], Any], instance: "BaseProblem", args: Tuple[Any, ...], kwargs: Mapping[str, Any]
) -> Any:
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
    if instance._problems is None:
        raise RuntimeError("TODO: Run prepare.")
    return wrapped(*args, **kwargs)
