import numpy as np
import numpy.typing as npt


# adapted from https://github.com/broadinstitute/wot/blob/master/notebooks/Notebook-2-compute-transport-maps.ipynb
def logistic(x: npt.ArrayLike, L: float, k: float, center: float=0) -> npt.ArrayLike:
    return L / (1 + np.exp(-k * (x - center)))

def gen_logistic(p: npt.ArrayLike, beta_max: float, beta_min: float, center: float, width: float) -> npt.ArrayLike:
    return beta_min + logistic(p, L=beta_max - beta_min, k=4 / width, center=center)

def beta(p: npt.ArrayLike, beta_max: float =1.7, beta_min: float=0.3, center: float=0.25, width: float=0.5, **kwargs) -> npt.ArrayLike:
    return gen_logistic(p, beta_max, beta_min, center, width)

def delta(a: npt.ArrayLike, delta_max: float=1.7, delta_min: float=0.3, center: float=0.1, width: float=0.2, **kwargs) -> npt.ArrayLike:
    return gen_logistic(a, delta_max, delta_min, center, width)
