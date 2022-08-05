from types import MappingProxyType
from typing import Any, Dict, Type, Tuple, Callable, Iterable, Optional, TYPE_CHECKING
from functools import partial, update_wrapper

from typing_extensions import Literal
import pandas as pd

from anndata import AnnData

from moscot._types import Filter_t
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
    annotations: Optional[Iterable[Any]] = None,
    aggregation_mode: Literal["annotation", "cell"] = AggregationMode.ANNOTATION,  # type: ignore[assignment]
) -> Iterable[Any]:
    if aggregation_mode == AggregationMode.ANNOTATION:  # type: ignore[comparison-overlap]
        if TYPE_CHECKING:  # checked in _check_argument_compatibility_cell_transition(
            assert annotations is not None
        annotations_verified = set(annotations).intersection(set(df[annotation_key].cat.categories))
        if not len(annotations_verified):
            raise ValueError(f"TODO: None of {annotations} found in distribution corresponding to {annotation_key}.")
        return annotations_verified
    return [None]


def _check_argument_compatibility_cell_transition(
    key: Optional[str] = None,
    other_key: Optional[str] = None,
    other_adata: Optional[str] = None,
    source_annotation: Filter_t = None,
    target_annotation: Filter_t = None,
    aggregation_mode: Literal["annotation", "cell"] = AggregationMode.ANNOTATION,  # type: ignore[assignment]
    forward: bool = False,
    **_: Any,
) -> None:
    if key is None and other_adata is None:
        raise ValueError("TODO: distributions cannot be inferred from `adata` due to missing obs keys.")
    if (forward and target_annotation is None) or (not forward and source_annotation is None):
        raise ValueError("TODO: obs column according to which is grouped is required.")
    if (AggregationMode(aggregation_mode) == AggregationMode.ANNOTATION) and (
        source_annotation is None or target_annotation is None
    ):
        raise ValueError("TODO: If `aggregation_mode` is `annotation` an `adata.obs` column must be provided.")


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
    arg: Filter_t = None,
) -> Tuple[Optional[str], Optional[Iterable[Any]]]:
    if arg is None:
        return (None, None)
    if isinstance(arg, str):
        if arg not in adata.obs:
            raise KeyError(f"TODO. {arg} not in adata.obs.columns")
        return arg, adata.obs[arg].cat.categories
    if isinstance(arg, dict):
        if len(arg) > 1:
            raise ValueError(f"Invalid dictionary length: `{len(arg)}` expected 1. ")
        key, val = next(iter(arg.items()))
        if not set(val).issubset(adata.obs[key].cat.categories):
            raise ValueError(f"Not all values {val} could be found in `adata.obs[{key}]`.")
        return key, val
    raise TypeError("TODO: `arg` must be of type `str` or `dict`")


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
    source_key: Optional[Any] = None,  # TODO(@MUCDK) using `K` induces circular import, resolve
    target_key: Optional[Any] = None,  # TODO(@MUCDK) using `K` induces circular import, resolve
) -> Tuple[Any, Any]:  # TODO(@MUCDK) using `K` induces circular import, resolve
    if source_key is not None and target_key is not None:
        return (source_key, target_key)
    elif source_key is None and target_key is not None:
        return ("src", target_key)  # TODO(@MUCDK) make package constant
    elif source_key is not None and target_key is None:
        return (source_key, "ref")  # TODO(@MUCDK) make package constant
    return ("src", "ref")
