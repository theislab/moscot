import numpy as np


# adapted from https://github.com/broadinstitute/wot/blob/master/notebooks/Notebook-2-compute-transport-maps.ipynb
# TODO(@MUCDK): check which variables are redundant.
def logistic(x, L, k, x0=0):
    f = L / (1 + np.exp(-k * (x - x0)))
    return f


def gen_logistic(p, beta_max, beta_min, pmax, pmin, center, width):
    return beta_min + logistic(p, L=beta_max - beta_min, k=4 / width, x0=center)


def beta(p, beta_max=1.7, beta_min=0.3, pmax=1.0, pmin=-0.5, center=0.25, **kwargs):
    return gen_logistic(p, beta_max, beta_min, pmax, pmin, center, width=0.5)


def delta(a, delta_max=1.7, delta_min=0.3, amax=0.5, amin=-0.4, center=0.1, **kwargs):
    return gen_logistic(a, delta_max, delta_min, amax, amin, center, width=0.2)
