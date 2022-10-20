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
SinkhornInitializer_t = Optional[Literal["default", "gaussian", "sorting"]]
QuadInitializer_t = Optional[Literal["random", "rank2", "k-means", "generalized-k-means"]]
Initializer_t = Union[SinkhornInitializer_t, QuadInitializer_t]
ProblemStage_t = Literal["initialized", "prepared", "solved"]

# TODO(michalk8): autogenerate from the enums
ScaleCost_t = Optional[Union[float, Literal["mean", "max_cost", "max_bound", "max_norm", "median"]]]
