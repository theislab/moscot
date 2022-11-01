from types import MappingProxyType
from typing import Any, Dict, List, Type, Tuple, Literal, Callable, Optional, TYPE_CHECKING
from functools import partial, update_wrapper
import inspect

import pandas as pd

from anndata import AnnData

from moscot._types import Str_Dict_t
from moscot._constants._constants import AggregationMode

__all__ = [
    "attributedispatch",
]

Callback = Callable[..., Any]


def attributedispatch(func: Optional[Callback] = None, attr: Optional[str] = None) -> Callback:
    """Dispatch a function based on the first value."""

    def dispatch(value: Type[Any]) -> Callback:
        for typ in value.mro():
            if typ in registry:
                return registry[typ]
        return func  # type: ignore[return-value]

    def register(value: Type[Any], func: Optional[Callback] = None) -> Callback:
        if func is None:
            return lambda f: register(value, f)
        registry[value] = func
        return func

    def wrapper(instance: Any, *args: Any, **kwargs: Any) -> Any:
        typ = type(getattr(instance, str(attr)))
        return dispatch(typ)(instance, *args, **kwargs)

    if func is None:
        return partial(attributedispatch, attr=attr)

    registry: Dict[Type[Any], Callback] = {}
    wrapper.register = register  # type: ignore[attr-defined]
    wrapper.dispatch = dispatch  # type: ignore[attr-defined]
    wrapper.registry = MappingProxyType(registry)  # type: ignore[attr-defined]
    update_wrapper(wrapper, func)

    return wrapper


def _validate_annotations_helper(
    df: pd.DataFrame,
    annotation_key: Optional[str] = None,
    annotations: Optional[List[Any]] = None,
    aggregation_mode: Literal["annotation", "cell"] = "annotation",
) -> List[Any]:
    if aggregation_mode == AggregationMode.ANNOTATION:
        if TYPE_CHECKING:  # checked in _check_argument_compatibility_cell_transition(
            assert annotations is not None
        annotations_verified = set(df[annotation_key].cat.categories).intersection(set(annotations))
        if not len(annotations_verified):
            raise ValueError(f"None of `{annotations}` found in the distribution corresponding to `{annotation_key}`.")
        return list(annotations_verified)
    return [None]


def _check_argument_compatibility_cell_transition(
    source_annotation: Str_Dict_t,
    target_annotation: Str_Dict_t,
    key: Optional[str] = None,
    # TODO(MUCDK): unused variable
    other_key: Optional[str] = None,
    other_adata: Optional[AnnData] = None,
    aggregation_mode: Literal["annotation", "cell"] = "annotation",
    forward: bool = False,
    **_: Any,
) -> None:
    if key is None and other_adata is None:
        raise ValueError("Unable to infer distributions, missing `adata` and `key`.")
    if forward and target_annotation is None:
        raise ValueError("No target annotation provided.")
    if not forward and source_annotation is None:
        raise ValueError("No source annotation provided.")
    if (AggregationMode(aggregation_mode) == AggregationMode.ANNOTATION) and (
        source_annotation is None or target_annotation is None
    ):
        raise ValueError(
            f"If aggregation mode is `{AggregationMode.ANNOTATION!r}`, "
            f"source or target annotation in `adata.obs` must be provided."
        )


def _get_df_cell_transition(
    adata: AnnData,
    key: Optional[str] = None,
    key_value: Optional[Any] = None,
    annotation_key: Optional[str] = None,
) -> pd.DataFrame:
    if key is None:
        return adata.obs[[annotation_key]].copy()
    return adata[adata.obs[key] == key_value].obs[[annotation_key]].copy()


def _validate_args_cell_transition(
    adata: AnnData,
    arg: Str_Dict_t,
) -> Tuple[str, List[Any], Optional[List[str]]]:
    if isinstance(arg, str):
        try:
            return arg, adata.obs[arg].cat.categories, None
        except KeyError:
            raise KeyError(f"Unable to fetch data from `adata.obs[{arg!r}]`.") from None
        except AttributeError:
            raise AttributeError(f"Data in `adata.obs[{arg!r}]` is not categorical.") from None

    if isinstance(arg, dict):
        if len(arg) != 1:
            raise ValueError(f"Expected dictionary of length `1`, found `{len(arg)}`.")
        key, val = next(iter(arg.items()))
        if not set(val).issubset(adata.obs[key].cat.categories):
            raise ValueError(f"Not all values `{val}` are present in `adata.obs[{key!r}]`.")
        return key, val, val

    raise TypeError(f"Expected argument to be either `str` or `dict`, found `{type(arg)}`.")


def _get_cell_indices(
    adata: AnnData,
    key: Optional[str] = None,
    key_value: Optional[Any] = None,
) -> pd.Index:
    if key is None:
        return adata.obs.index
    return adata[adata.obs[key] == key_value].obs.index


def _get_categories_from_adata(
    adata: AnnData,
    key: Optional[str] = None,
    key_value: Optional[Any] = None,
    annotation_key: Optional[str] = None,
) -> pd.Series:
    if key is None:
        return adata.obs[annotation_key]
    return adata[adata.obs[key] == key_value].obs[annotation_key]


def _get_problem_key(
    source: Optional[Any] = None,  # TODO(@MUCDK) using `K` induces circular import, resolve
    target: Optional[Any] = None,  # TODO(@MUCDK) using `K` induces circular import, resolve
) -> Tuple[Any, Any]:  # TODO(@MUCDK) using `K` induces circular import, resolve
    if source is not None and target is not None:
        return (source, target)
    elif source is None and target is not None:
        return ("src", target)  # TODO(@MUCDK) make package constant
    elif source is not None and target is None:
        return (source, "ref")  # TODO(@MUCDK) make package constant
    return ("src", "ref")


def _order_transition_matrix_helper(
    tm: pd.DataFrame,
    rows_verified: List[str],
    cols_verified: List[str],
    row_order: Optional[List[str]],
    col_order: Optional[List[str]],
) -> pd.DataFrame:
    if col_order is not None:
        tm = tm[[col for col in col_order if col in cols_verified]]
    tm = tm.T
    if row_order is not None:
        return tm[[row for row in row_order if row in rows_verified]]
    return tm


def _order_transition_matrix(
    tm: pd.DataFrame,
    source_annotations_verified: List[str],
    target_annotations_verified: List[str],
    source_annotations_ordered: Optional[List[str]],
    target_annotations_ordered: Optional[List[str]],
    forward: bool,
) -> pd.DataFrame:

    if target_annotations_ordered is not None or source_annotations_ordered is not None:
        if forward:
            tm = _order_transition_matrix_helper(
                tm=tm,
                rows_verified=source_annotations_verified,
                cols_verified=target_annotations_verified,
                row_order=source_annotations_ordered,
                col_order=target_annotations_ordered,
            )
        else:
            tm = _order_transition_matrix_helper(
                tm=tm,
                rows_verified=target_annotations_verified,
                cols_verified=source_annotations_verified,
                row_order=target_annotations_ordered,
                col_order=source_annotations_ordered,
            )
        return tm.T if forward else tm
    elif target_annotations_verified == source_annotations_verified:
        annotations_ordered = tm.columns.sort_values()
        if forward:
            tm = _order_transition_matrix_helper(
                tm=tm,
                rows_verified=source_annotations_verified,
                cols_verified=target_annotations_verified,
                row_order=annotations_ordered,
                col_order=annotations_ordered,
            )
        else:
            tm = _order_transition_matrix_helper(
                tm=tm,
                rows_verified=target_annotations_verified,
                cols_verified=source_annotations_verified,
                row_order=annotations_ordered,
                col_order=annotations_ordered,
            )
        return tm.T if forward else tm
    return tm if forward else tm.T


def _filter_kwargs(*funcs: Callable[..., Any], **kwargs: Any) -> Dict[str, Any]:
    res = {}
    for func in funcs:
        params = inspect.signature(func).parameters
        res.update({k: v for k, v in kwargs.items() if k in params})
    return res
