import types
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from moscot._types import CostFn_t, CostKwargs_t


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
        if joint_attr.get("tag", None) == "cost_matrix" and (
            len(joint_attr) == 2 or kwargs.get("attr", None) == "obsp"
        ):
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
    cost: Optional[Union[CostFn_t, Mapping[str, CostFn_t]]] = None,
    cost_kwargs: CostKwargs_t = types.MappingProxyType({}),
    **kwargs: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    xy, x, y = dict(xy), dict(x), dict(y)
    if cost is None:
        return xy, x, y
    if isinstance(cost, str):  # if cost is a str, we use it in all terms
        if (xy or "xy_callback" in kwargs) and "cost" not in xy:
            xy["x_cost"] = xy["y_cost"] = cost
        if (x or "x_callback" in kwargs) and "cost" not in x:
            x["cost"] = cost
        if (y or "y_callback" in kwargs) and "cost" not in y:
            y["cost"] = cost
    elif isinstance(cost, Mapping):  # if cost is a dict, the cost is specified for each term
        if (xy or "xy_callback" in kwargs) and "cost" not in xy:
            xy["x_cost"] = xy["y_cost"] = cost["xy"]
        if (x or "x_callback" in kwargs) and "cost" not in x:
            x["cost"] = cost["x"]
        if (y or "y_callback" in kwargs) and "cost" not in y:
            y["cost"] = cost["y"]
    else:
        raise TypeError(f"Expected `cost` to be either `str` or `dict`, found `{type(cost)}`.")
    if (xy or "xy_callback" in kwargs) and cost_kwargs:  # distribute the cost_kwargs, possibly explicit to x/y/xy-term
        # extract cost_kwargs explicit to xy-term if possible
        items = cost_kwargs["xy"].items() if "xy" in cost_kwargs else cost_kwargs.items()
        for k, v in items:
            xy[f"x_{k}"] = xy[f"y_{k}"] = v
    if (x or "x_callback" in kwargs) and cost_kwargs:  # extract cost_kwargs explicit to x-term if possible
        x.update(cost_kwargs.get("x", cost_kwargs))  # type:ignore[call-overload]
    if (y or "y_callback" in kwargs) and cost_kwargs:  # extract cost_kwargs explicit to y-term if possible
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
