import types
from typing import Any, Dict, Literal, Mapping, Optional, Tuple, Union

from moscot._types import CostKwargs_t, OttCostFnMap_t
from moscot.base.problems.compound_problem import Callback_t


def handle_joint_attr(
    joint_attr: Optional[Union[str, Mapping[str, Any]]], kwargs: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if joint_attr is None:
        if "xy_callback" not in kwargs:
            kwargs["xy_callback"] = "local-pca"
        kwargs.setdefault("xy_callback_kwargs", {})
        return {}, kwargs
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

        # if this is True we have custom cost matrix or moscot cost - in this case we have a custom cost matrix
        if joint_attr.get("tag", None) == "cost_matrix" and (len(joint_attr) == 2 or kwargs.get("attr") == "obsp"):
            joint_attr.setdefault("cost", "custom")
            joint_attr.setdefault("attr", "obsp")
            kwargs["xy_callback"] = "cost-matrix"
            kwargs.setdefault("xy_callback_kwargs", {"key": joint_attr["key"]})
        kwargs.setdefault("xy_callback_kwargs", {})
        return joint_attr, kwargs
    raise TypeError(f"Expected `joint_attr` to be either `str` or `dict`, found `{type(joint_attr)}`.")


def handle_cost(
    xy: Mapping[str, Any] = types.MappingProxyType({}),
    x: Mapping[str, Any] = types.MappingProxyType({}),
    y: Mapping[str, Any] = types.MappingProxyType({}),
    cost: Optional[OttCostFnMap_t] = None,
    cost_kwargs: CostKwargs_t = types.MappingProxyType({}),
    xy_callback: Optional[Union[Literal["local-pca"], Callback_t]] = None,
    x_callback: Optional[Union[Literal["local-pca"], Callback_t]] = None,
    y_callback: Optional[Union[Literal["local-pca"], Callback_t]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    xy, x, y = dict(xy), dict(x), dict(y)
    if cost is None:
        return xy, x, y

    # cost candidates to set
    # if cost_candidates["x"] is True, x["cost"] is set
    cost_candidates = {
        "xy": (xy or xy_callback) and "cost" not in xy,
        "x": (x or x_callback) and "cost" not in x,
        "y": (y or y_callback) and "cost" not in y,
    }
    if isinstance(cost, Mapping):
        cost_candidates = {k: cost[k] for k, v in cost_candidates.items() if v}  # type:ignore[index,misc]
    elif isinstance(cost, str):
        cost_candidates = {k: cost for k, v in cost_candidates.items() if v}  # type:ignore[misc]
    else:
        raise TypeError(f"Expected `cost` to be either `str` or `dict`, found `{type(cost)}`.")

    # set cost
    if "xy" in cost_candidates:
        xy["x_cost"] = xy["y_cost"] = cost_candidates["xy"]
    if "x" in cost_candidates:
        x["cost"] = cost_candidates["x"]
    if "y" in cost_candidates:
        y["cost"] = cost_candidates["y"]

    if cost_kwargs:
        if "xy" in cost_candidates:
            items = cost_kwargs["xy"].items() if "xy" in cost_kwargs else cost_kwargs.items()
            for k, v in items:
                xy[f"x_{k}"] = xy[f"y_{k}"] = v
        if "x" in cost_candidates:
            x.update(cost_kwargs.get("x", cost_kwargs))  # type:ignore[call-overload]
        if "y" in cost_candidates:
            y.update(cost_kwargs.get("y", cost_kwargs))  # type:ignore[call-overload]
    return xy, x, y


def pop_callbacks_compound_prepare(kwargs: Dict[str, Any]) -> Tuple[Any, ...]:
    """
    Pop callbacks from kwargs and return x, y, xy callbacks and their kwargs,
    then reference and subset respectively. For use before `CompoundProblem.prepare`.
    """  # noqa: D205
    cb = pop_callbacks(kwargs)
    kws = pop_callback_kwargs(kwargs)
    others = pop_reference_subset(kwargs)
    return (*cb, *kws, *others)


def pop_callbacks(kwargs: Dict[str, Any]) -> Tuple[Any, ...]:
    """Pop callbacks from kwargs and return x, y, xy callbacks respectively."""  # noqa: D205
    cb_keys = ("x", "y", "xy")
    return tuple(kwargs.pop(k + "_callback", None) for k in cb_keys)


def pop_callback_kwargs(kwargs: Dict[str, Any]) -> Tuple[Any, ...]:
    """Pop callbacks from kwargs and return x, y, xy callback kwargs respectively."""  # noqa: D205
    cb_keys = ("x", "y", "xy")
    return tuple(kwargs.pop(k + "_callback_kwargs", {}) for k in cb_keys)


def pop_reference_subset(kwargs: Dict[str, Any]) -> Tuple[Any, Any]:
    """Pop reference and subset from kwargs and return them respectively."""  # noqa: D205
    return kwargs.pop("reference", None), kwargs.pop("subset", None)
