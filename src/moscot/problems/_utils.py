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
    if cost is None:
        return xy, x, y  # type:ignore[return-value]
    if isinstance(cost, str):
        if xy is not None and "cost" not in xy:
            xy = dict(xy)
            xy["x_cost"] = cost
            xy["y_cost"] = cost
        if not (x is None or len(x) == 0) and "cost" not in x:
            x = dict(x)
            x["cost"] = cost
        if not (y is None or len(y) == 0) and "cost" not in y:
            y = dict(y)
            y["cost"] = cost
    elif isinstance(cost, Mapping):
        if xy is not None and "cost" not in xy:
            xy = dict(xy)
            xy["x_cost"] = cost["xy"]
            xy["y_cost"] = cost["xy"]
        if not (x is None or len(x) == 0) and "cost" not in x:
            x = dict(x)
            x["cost"] = cost["x"]
        if not (y is None or len(y) == 0) and "cost" not in y:
            y = dict(y)
            y["cost"] = cost["y"]
    else:
        raise TypeError(type(cost))
    if xy is not None and len(cost_kwargs):
        xy = dict(xy)
        if "xy" in cost_kwargs:
            k, v = next(iter(cost_kwargs["xy"].items()))
        else:
            k, v = next(iter(cost_kwargs.items()))
        xy["x_" + k] = v
        xy["y_" + k] = v
        # xy.update(cost_kwargs if "xy" not in cost_kwargs else cost_kwargs["xy"])
    if not (x is None or len(x) == 0) and len(cost_kwargs):
        x = dict(x)
        x.update(cost_kwargs if "x" not in cost_kwargs else cost_kwargs["x"])
    if not (y is None or len(y) == 0) and len(cost_kwargs):
        y = dict(y)
        y.update(cost_kwargs if "y" not in cost_kwargs else cost_kwargs["y"])
    return xy, x, y  # type:ignore[return-value]
