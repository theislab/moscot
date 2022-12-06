from typing import Any

import numpy as np

from moscot._types import ArrayLike


# adapted from https://github.com/broadinstitute/wot/blob/master/notebooks/Notebook-2-compute-transport-maps.ipynb
def logistic(x: ArrayLike, L: float, k: float, center: float = 0) -> ArrayLike:
    """Logistic function."""
    return L / (1 + np.exp(-k * (x - center)))


def gen_logistic(p: ArrayLike, beta_max: float, beta_min: float, center: float, width: float) -> ArrayLike:
    """Shifted logistic function."""
    return beta_min + logistic(p, L=beta_max - beta_min, k=4 / width, center=center)


def beta(
    p: ArrayLike,
    beta_max: float = 1.7,
    beta_min: float = 0.3,
    center: float = 0.25,
    width: float = 0.5,
    **_: Any,
) -> ArrayLike:
    """Birth process."""
    return gen_logistic(p, beta_max, beta_min, center, width)


def delta(
    a: ArrayLike,
    delta_max: float = 1.7,
    delta_min: float = 0.3,
    center: float = 0.1,
    width: float = 0.2,
    **kwargs: Any,
) -> ArrayLike:
    """Death process."""
    return gen_logistic(a, delta_max, delta_min, center, width)
