from typing import Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
import warnings

from numpy import typing as npt
import numpy as np


def _warn_not_close(
    actual: Optional[npt.ArrayLike],
    expected: Optional[npt.ArrayLike],
    *,
    kind: Literal["source", "target"],
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> None:
    try:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False, verbose=False)
    except AssertionError:
        # TODO(michalk8): parse how many elements passed the check
        msg = f"{kind.capitalize()} marginals are not satisfied within rtol={rtol}, atol={atol}."
        warnings.warn(msg)
