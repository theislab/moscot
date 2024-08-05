import types
from typing import Any, Dict, Literal, Mapping, Optional, Tuple, Union

from moscot._types import CostFn_t, CostKwargs_t, OttCostFnMap_t
from moscot.base.problems.compound_problem import Callback_t


def _validate_joint_attr(joint_attr: Optional[Union[str, Mapping[str, Any]]]) -> None:
    if joint_attr is not None and not isinstance(joint_attr, (str, Mapping)):
        raise TypeError(f"Expected `joint_attr` to be either `str` or `dict`, found `{type(joint_attr)}`.")


def _handle_none_joint_attr(
    xy_callback: Optional[Union[Literal["local-pca"], Callback_t]] = None,
    xy_callback_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
) -> Tuple[Dict[str, Any], Union[Literal["local-pca"], Callback_t], Mapping[str, Any]]:
    if xy_callback is None:
        xy_callback = "local-pca"
    return {}, xy_callback, xy_callback_kwargs


def _handle_string_joint_attr(
    joint_attr: str,
    xy_callback: Optional[Union[Literal["local-pca"], Callback_t]] = None,
    xy_callback_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
) -> Tuple[Dict[str, Any], Union[Literal["local-pca"], Callback_t], Mapping[str, Any]]:
    xy = {
        "x_attr": "obsm",
        "x_key": joint_attr,
        "y_attr": "obsm",
        "y_key": joint_attr,
    }
    return xy, xy_callback, xy_callback_kwargs  # type: ignore[return-value]


def _handle_mapping_joint_attr(
    joint_attr: Mapping[str, Any],
    xy_callback: Optional[Union[Literal["local-pca"], Callback_t]] = None,
    xy_callback_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
) -> Tuple[Dict[str, Any], Union[Literal["local-pca"], Callback_t], Dict[str, Any]]:
    joint_attr = dict(joint_attr)
    xy_callback_kwargs = dict(xy_callback_kwargs)
    if "attr" in joint_attr and joint_attr["attr"] == "X":
        return {"x_attr": "X", "y_attr": "X"}, xy_callback, xy_callback_kwargs  # type: ignore[return-value]
    if "attr" in joint_attr and joint_attr["attr"] == "obsm":
        if "key" not in joint_attr:
            raise KeyError("`key` must be provided when `attr` is `obsm`.")
        xy = {
            "x_attr": "obsm",
            "x_key": joint_attr["key"],
            "y_attr": "obsm",
            "y_key": joint_attr["key"],
        }
        return xy, xy_callback, xy_callback_kwargs  # type: ignore[return-value]

    if joint_attr.get("tag", None) == "cost_matrix" and (len(joint_attr) == 2 or joint_attr.get("attr") == "obsp"):
        joint_attr.setdefault("cost", "custom")
        joint_attr.setdefault("attr", "obsp")
        xy_callback = "cost-matrix"
        xy_callback_kwargs = xy_callback_kwargs or {}
        xy_callback_kwargs["key"] = joint_attr["key"]
    return joint_attr, xy_callback, xy_callback_kwargs  # type: ignore[return-value]


def handle_joint_attr(
    joint_attr: Optional[Union[str, Mapping[str, Any]]],
    xy_callback: Optional[Union[Literal["local-pca"], Callback_t]] = None,
    xy_callback_kwargs: Mapping[str, Any] = types.MappingProxyType({}),
) -> Tuple[Mapping[str, Any], Union[Literal["local-pca"], Callback_t], Mapping[str, Any]]:
    _validate_joint_attr(joint_attr)

    if joint_attr is None:
        return _handle_none_joint_attr(xy_callback, xy_callback_kwargs)
    if isinstance(joint_attr, str):
        return _handle_string_joint_attr(joint_attr, xy_callback, xy_callback_kwargs)
    if isinstance(joint_attr, Mapping):  # input mapping does not distinguish between x and y as it's a shared space
        return _handle_mapping_joint_attr(joint_attr, xy_callback, xy_callback_kwargs)
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


def handle_conditional_attr(conditional_attr: Optional[Union[str, Mapping[str, Any]]]) -> Dict[str, Any]:
    if isinstance(conditional_attr, str):
        conditional_attr = {"attr": "obsm", "key": conditional_attr}
    elif isinstance(conditional_attr, Mapping):
        conditional_attr = dict(conditional_attr)
        if "attr" not in conditional_attr:
            raise KeyError("`attr` must be provided when `conditional_attr` is a mapping.")
        if conditional_attr["attr"] == "X":
            conditions_attr = "X"
            conditions_key = None
        else:
            if "key" not in conditional_attr:
                raise KeyError("`key` must be provided when `attr` is not `X`.")
            conditions_attr = conditional_attr["attr"]
            conditions_key = conditional_attr["key"]
    else:
        raise TypeError("Expected `conditional_attr` to be either `str` or `dict`.")
    return {"conditions_attr": conditions_attr, "conditions_key": conditions_key}


def handle_joint_attr_tmp(
    joint_attr: Union[str, Mapping[str, Any]], kwargs: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if isinstance(joint_attr, str):
        xy = {
            "xy_attr": "obsm",
            "xy_key": joint_attr,
        }
        return xy, kwargs
    if isinstance(joint_attr, Mapping):  # input mapping does not distinguish between x and y as it's a shared space
        joint_attr = dict(joint_attr)
        if "attr" in joint_attr and joint_attr["attr"] == "X":  # we have a point cloud
            return {"xy_attr": "X"}, kwargs
        if "attr" in joint_attr and joint_attr["attr"] == "obsm":  # we have a point cloud
            if "key" not in joint_attr:
                raise KeyError("`key` must be provided when `attr` is `obsm`.")
            xy = {
                "xy_attr": "obsm",
                "xy_key": joint_attr["key"],
            }
            return xy, kwargs

    raise TypeError(f"Expected `joint_attr` to be either `str` or `dict`, found `{type(joint_attr)}`.")


def handle_cost_tmp(
    xy: Mapping[str, Any] = types.MappingProxyType({}),
    xx: Mapping[str, Any] = types.MappingProxyType({}),
    cost: Optional[Union[CostFn_t, Mapping[str, CostFn_t]]] = None,
    cost_kwargs: CostKwargs_t = types.MappingProxyType({}),
    **_: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    xy, xx = dict(xy), dict(xx)
    if cost is None:
        return xy, xx
    if isinstance(cost, str):  # if cost is a str, we use it in all terms
        if xy and "cost" not in xy:
            xy["xy_cost"] = cost
        if xx and "cost" not in xx:
            xx["xy_cost"] = cost
    elif isinstance(cost, Mapping):  # if cost is a dict, the cost is specified for each term
        if xy and ("xy_cost" not in xy or "xx_cost" not in xy):
            xy["xy_cost"] = cost["xy"]
        if xx and "cost" not in xx:
            xx["xx_cost"] = cost["xx_cost"]
    else:
        raise TypeError(f"Expected `cost` to be either `str` or `dict`, found `{type(cost)}`.")
    if xy and cost_kwargs:  # distribute the cost_kwargs, possibly explicit to x/y/xy-term
        # extract cost_kwargs explicit to xy-term if possible
        items = cost_kwargs["xy"].items() if "xy" in cost_kwargs else cost_kwargs.items()
        for k, v in items:
            xy[f"xy_{k}"] = xy[f"xy_{k}"] = v
    if xx and cost_kwargs:  # extract cost_kwargs explicit to x-term if possible
        xx.update(cost_kwargs.get("xx", cost_kwargs))  # type:ignore[call-overload]
    return xy, xx
