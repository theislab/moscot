import os
from typing import Any, Literal, Mapping, Optional, Sequence, Union

import numpy as np
from ott.initializers.linear.initializers import SinkhornInitializer
from ott.initializers.linear.initializers_lr import LRInitializer
from ott.initializers.quadratic.initializers import BaseQuadraticInitializer

# TODO(michalk8): polish

try:
    from numpy.typing import DTypeLike, NDArray

    ArrayLike = NDArray[np.float64]
except (ImportError, TypeError):
    ArrayLike = np.ndarray  # type: ignore[misc]
    DTypeLike = np.dtype  # type: ignore[misc]

ProblemKind_t = Literal["linear", "quadratic", "unknown"]
Numeric_t = Union[int, float]  # type of `time_key` arguments
Filter_t = Optional[Union[str, Mapping[str, Sequence[Any]]]]  # type how to filter adata
Str_Dict_t = Optional[Union[str, Mapping[str, Sequence[Any]]]]  # type for `cell_transition`
SinkhornInitializerTag_t = Literal["default", "gaussian", "sorting"]
LRInitializerTag_t = Literal["random", "rank2", "k-means", "generalized-k-means"]

LRInitializer_t = Optional[Union[LRInitializer, LRInitializerTag_t]]
SinkhornInitializer_t = Optional[Union[SinkhornInitializer, SinkhornInitializerTag_t]]
QuadInitializer_t = Optional[Union[BaseQuadraticInitializer]]

Initializer_t = Union[SinkhornInitializer_t, QuadInitializer_t, LRInitializer_t]
ProblemStage_t = Literal["prepared", "solved"]
Device_t = Union[Literal["cpu", "gpu", "tpu"], str]

# TODO(michalk8): autogenerate from the enums
ScaleCost_t = Union[float, Literal["mean", "max_cost", "max_bound", "max_norm", "median"]]
OttCostFn_t = Literal[
    "euclidean",
    "sq_euclidean",
    "cosine",
    "pnorm_p",
    "sq_pnorm",
    "cosine",
    "geodesic",
]
OttCostFnMap_t = Union[OttCostFn_t, Mapping[Literal["xy", "x", "y"], OttCostFn_t]]
GenericCostFn_t = Literal["barcode_distance", "leaf_distance", "custom"]
CostFn_t = Union[str, GenericCostFn_t, OttCostFn_t]
CostFnMap_t = Union[Union[OttCostFn_t, GenericCostFn_t], Mapping[str, Union[OttCostFn_t, GenericCostFn_t]]]
PathLike = Union[os.PathLike, str]  # type: ignore[type-arg]
Policy_t = Literal[
    "sequential",
    "star",
    "external_star",
    "explicit",
    "triu",
    "tril",
]
CostKwargs_t = Union[Mapping[str, Any], Mapping[Literal["x", "y", "xy"], Mapping[str, Any]]]
