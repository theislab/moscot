from typing import Any, Dict, Tuple, Union, Mapping, Callable, Optional, TYPE_CHECKING

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
        raise RuntimeError("Run `.solve()` first.")
    if isinstance(instance, BaseCompoundProblem) and instance.solutions is None:
        raise RuntimeError("Run `.solve()` first.")
    return wrapped(*args, **kwargs)


@wrapt.decorator
def require_prepare(
    wrapped: Callable[[Any], Any], instance: "BaseCompoundProblem", args: Tuple[Any, ...], kwargs: Mapping[str, Any]
) -> Any:
    """Check whether problem has been prepared."""
    if instance.problems is None:
        raise RuntimeError("Run `.prepare()` first.")
    return wrapped(*args, **kwargs)


@wrapt.decorator
def wrap_prepare(
    wrapped: Callable[[Any], Any], instance: "BaseProblem", args: Tuple[Any, ...], kwargs: Mapping[str, Any]
) -> Any:
    """Check and update the state when preparing :class:`moscot.problems.base.OTProblem`."""
    from moscot._constants._constants import ProblemStage
    from moscot.problems.base._base_problem import ProblemKind  # TODO: move ENUMs to this file

    instance = wrapped(*args, **kwargs)
    if instance.problem_kind == ProblemKind.UNKNOWN:
        raise RuntimeError("Problem kind was not set after running `.prepare()`.")
    instance._stage = ProblemStage.PREPARED
    return instance


@wrapt.decorator
def wrap_solve(
    wrapped: Callable[[Any], Any], instance: "BaseProblem", args: Tuple[Any, ...], kwargs: Mapping[str, Any]
) -> Any:
    """Check and update the state when solving :class:`moscot.problems.base.OTProblem`."""
    from moscot._constants._constants import ProblemStage

    if instance.stage not in (ProblemStage.PREPARED, ProblemStage.SOLVED):
        raise RuntimeError(
            f"Expected problem's stage to be either `'prepared'` or `'solved'`, found `{instance.stage!r}`."
        )
    instance = wrapped(*args, **kwargs)
    instance._stage = ProblemStage.SOLVED
    return instance


def handle_joint_attr(
    joint_attr: Optional[Union[str, Mapping[str, Any]]], kwargs: Any
) -> Tuple[Optional[Union[str, Mapping[str, Any]]], Dict[str, Any]]:
    if joint_attr is None:
        if "callback" not in kwargs:
            kwargs["callback"] = "local-pca"
        else:
            kwargs["callback"] = kwargs["callback"]
        kwargs["callback_kwargs"] = {**kwargs.get("callback_kwargs", {}), **{"return_linear": True}}
        return None, kwargs
    if isinstance(joint_attr, str):
        xy = {
            "x_attr": "obsm",
            "x_key": joint_attr,
            "y_attr": "obsm",
            "y_key": joint_attr,
        }
        return xy, kwargs
    if isinstance(joint_attr, Mapping):
        return joint_attr, kwargs
    raise TypeError(f"Expected `joint_attr` to be either `str` or `dict`, found `{type(joint_attr)}`.")
