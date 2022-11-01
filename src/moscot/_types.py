from typing import Any, Union, Literal, Mapping, Optional, Sequence

import numpy as np

__all__ = ["ArrayLike", "DTypeLike", "Numeric_t"]


try:
    from numpy.typing import NDArray, DTypeLike

    ArrayLike = NDArray[np.float_]
except (ImportError, TypeError):
    ArrayLike = np.ndarray  # type: ignore[misc]
    DTypeLike = np.dtype  # type: ignore[misc]

Numeric_t = Union[int, float]  # type of `time_key` arguments
Filter_t = Optional[Union[str, Mapping[str, Sequence[Any]]]]  # type how to filter adata
Str_Dict_t = Union[str, Mapping[str, Sequence[Any]]]  # type for `cell_transition`
SinkFullRankInit = Union[Literal["default", "gaussian", "sorting"]]
LRInitializer_t = Literal["random", "rank2", "k-means", "generalized-k-means"]

SinkhornInitializer_t = Optional[Union[SinkFullRankInit, LRInitializer_t]]
QuadInitializer_t = Optional[LRInitializer_t]

Initializer_t = Union[SinkhornInitializer_t, LRInitializer_t]
ProblemStage_t = Literal["initialized", "prepared", "solved"]
Device_t = Literal["cpu", "gpu", "tpu"]

# TODO(michalk8): autogenerate from the enums
ScaleCost_t = Optional[Union[float, Literal["mean", "max_cost", "max_bound", "max_norm", "median"]]]
OttCostFn_t = Literal["euclidean", "sq_euclidean", "cosine", "bures", "unbalanced_bures"]
GenericCostFn_t = Literal["barcode_distance", "leaf_distance", "custom"]
CostFn_t = Union[str, GenericCostFn_t, OttCostFn_t]
