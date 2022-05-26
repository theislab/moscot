import numpy as np

__all__ = ["ArrayLike"]

try:
    from numpy.typing import NDArray

    ArrayLike = NDArray[np.float]
except (ImportError, TypeError):
    ArrayLike = np.ndarray  # type: ignore[misc]