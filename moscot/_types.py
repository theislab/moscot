from typing import Union

import numpy as np

__all__ = ["ArrayLike", "Numeric_t"]

try:
    from numpy.typing import NDArray, DTypeLike

    ArrayLike = NDArray[np.float]
except (ImportError, TypeError):
    ArrayLike = np.ndarray  # type: ignore[misc]
    DTypeLike = np.dtype

Numeric_t = Union[int, float]
