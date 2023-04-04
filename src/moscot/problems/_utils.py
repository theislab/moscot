import types
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from moscot._types import CostFn_t


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
    xy: Optional[Mapping[str, Any]] = None,
    x: Optional[Mapping[str, Any]] = None,
    y: Optional[Mapping[str, Any]] = None,
    cost: Optional[Union[CostFn_t, Mapping[str, CostFn_t]]] = None,
    cost_kwargs: Union[Mapping[str, Any], Mapping[str, Mapping[str, Any]]] = types.MappingProxyType({}),
    **_: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    x = {} if x is None else dict(x)
    y = {} if y is None else dict(y)
    xy = {} if xy is None else dict(xy)
    if cost is None:
        return xy if len(xy) > 0 else None, x if len(x) > 0 else None, y if len(y) > 0 else None
    if isinstance(cost, str):  # if cost is a str, we use it in all terms
        if len(xy) > 0 and "cost" not in xy:
            xy["x_cost"] = xy["y_cost"] = cost
        if len(x) > 0 and "cost" not in x:
            x["cost"] = cost
        if len(y) > 0 and "cost" not in y:
            y["cost"] = cost
    elif isinstance(cost, Mapping):  # if cost is a dict, the cost is specified for each term
        if len(xy) > 0 and "cost" not in xy:
            xy["x_cost"] = xy["y_cost"] = cost["xy"]
        if len(x) > 0 and "cost" not in x:
            x["cost"] = cost["x"]
        if len(y) > 0 and "cost" not in y:
            y["cost"] = cost["y"]
    else:
        raise TypeError(type(cost))
    if len(xy) > 0 and len(cost_kwargs):  # distribute the cost_kwargs, possibly explicit to x/y/xy-term
        if "xy" in cost_kwargs:
            k, v = next(iter(cost_kwargs["xy"].items()))  # extract cost_kwargs explicit to xy-term if possible
        else:
            k, v = next(iter(cost_kwargs.items()))
        xy["x_" + k] = xy["y_" + k] = v
    if len(x) > 0 and len(cost_kwargs):
        x.update(cost_kwargs.get("x", cost_kwargs))  # extract cost_kwargs explicit to x-term if possible
    if len(y) > 0 and len(cost_kwargs):
        y.update(cost_kwargs.get("y", cost_kwargs))  # extract cost_kwargs explicit to y-term if possible
    return xy if len(xy) > 0 else None, x if len(x) > 0 else None, y if len(y) > 0 else None
