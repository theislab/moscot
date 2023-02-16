from typing import Any, Dict, Tuple, Union, Mapping, Callable, Optional, TYPE_CHECKING

from moscot._types import CostFn_t

if TYPE_CHECKING:
    from moscot.problems.base import BaseProblem  # type: ignore[attr-defined]

import wrapt

__all__ = ["wrap_prepare", "wrap_solve"]


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
    joint_attr: Optional[Union[str, Mapping[str, Any]]], kwargs: Dict[str, Any]
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    if joint_attr is None:
        if "xy_callback" not in kwargs:
            kwargs["xy_callback"] = "local-pca"
        kwargs.setdefault("xy_callback_kwargs", {})
        return None, kwargs
    if isinstance(joint_attr, str):
        xy = {
            "x_attr": "obsm",
            "x_key": joint_attr,
            "y_attr": "obsm",
            "y_key": joint_attr,
        }
        return xy, kwargs
    if isinstance(joint_attr, Mapping):  # input mapping does not distinguish between x and y as it's a shared space
        joint_attr = dict(joint_attr)
        if "attr" in joint_attr and joint_attr["attr"] == "X":  # we have a point cloud
            return {"x_attr": "X", "y_attr": "X"}, kwargs
        if "attr" in joint_attr and joint_attr["attr"] == "obsm":  # we have a point cloud
            if "key" not in joint_attr:
                raise KeyError("`key` must be provided when `attr` is `obsm`.")
            xy = {
                "x_attr": "obsm",
                "x_key": joint_attr["key"],
                "y_attr": "obsm",
                "y_key": joint_attr["key"],
            }
            return xy, kwargs
        if joint_attr.get("tag", None) == "cost_matrix":  # if this is True we have custom cost matrix or moscot cost
            if len(joint_attr) == 2 or kwargs.get("attr", None) == "obsp":  # in this case we have a custom cost matrix
                joint_attr.setdefault("cost", "custom")
                joint_attr.setdefault("attr", "obsp")
                kwargs["xy_callback"] = "cost-matrix"
                kwargs.setdefault("xy_callback_kwargs", {"key": joint_attr["key"]})
        kwargs.setdefault("xy_callback_kwargs", {})
        return joint_attr, kwargs
    raise TypeError(f"Expected `joint_attr` to be either `str` or `dict`, found `{type(joint_attr)}`.")


def handle_cost(
    xy: Optional[Mapping[str, Any]] = None,
    x: Optional[Mapping[str, Any]] = None,
    y: Optional[Mapping[str, Any]] = None,
    cost: Optional[Union[CostFn_t, Mapping[str, CostFn_t]]] = None,
    **_: Any,
) -> Tuple[Optional[Mapping[str, Any]], Optional[Mapping[str, Any]], Optional[Mapping[str, Any]]]:
    if cost is None:
        return xy, x, y
    if isinstance(cost, str):
        if xy is not None and "cost" not in xy:
            xy = dict(xy)
            xy["cost"] = cost
        if x is not None and "cost" not in x:
            x = dict(x)
            x["cost"] = cost
        if y is not None and "cost" not in y:
            y = dict(y)
            y["cost"] = cost
        return xy, x, y
    if isinstance(cost, Mapping):
        if xy is not None and "cost" not in xy:
            xy = dict(xy)
            xy["cost"] = cost["xy"]
        if x is not None and "cost" not in x:
            x = dict(x)
            x["cost"] = cost["x"]
        if y is not None and "cost" not in y:
            y = dict(y)
            y["cost"] = cost["y"]
        return xy, x, y
