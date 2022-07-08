from typing import Union

import numpy as np

__all__ = ["ArrayLike", "DTypeLike", "Numeric_t"]

try:
    from numpy.typing import NDArray, DTypeLike

    ArrayLike = NDArray[np.float_]
except (ImportError, TypeError):
    ArrayLike = np.ndarray  # type: ignore[misc]
    DTypeLike = np.dtype  # type: ignore[misc]

Numeric_t = Union[int, float]  # for `time_key` arguments
