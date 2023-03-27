from typing import Any

import numpy as np

from moscot._types import ArrayLike


# adapted from https://github.com/broadinstitute/wot/blob/master/notebooks/Notebook-2-compute-transport-maps.ipynb
def logistic(x: ArrayLike, L: float, k: float, center: float = 0) -> ArrayLike:
    """Logistic function."""
    return L / (1 + np.exp(-k * (x - center)))


def gen_logistic(p: ArrayLike, sup: float, inf: float, center: float, width: float) -> ArrayLike:
    """Shifted logistic function."""
    return inf + logistic(p, L=sup - inf, k=4 / width, center=center)


def beta(
    p: ArrayLike,
    beta_max: float = 1.7,
    beta_min: float = 0.3,
    beta_center: float = 0.25,
    beta_width: float = 0.5,
    **_: Any,
) -> ArrayLike:
    """Birth process."""
    return gen_logistic(p, beta_max, beta_min, beta_center, beta_width)


def delta(
    a: ArrayLike,
    delta_max: float = 1.7,
    delta_min: float = 0.3,
    delta_center: float = 0.1,
    delta_width: float = 0.2,
    **_: Any,
) -> ArrayLike:
    """Death process."""
    return gen_logistic(a, delta_max, delta_min, delta_center, delta_width)
