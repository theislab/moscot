from typing import Union, Sequence, Any, Optional
from numpy.typing import ArrayLike
from anndata import AnnData
from moscot.tmp.solvers._data import Tag
from numpy.typing import ArrayLike
import numpy as np
from ott.geometry.costs import Euclidean, Cosine, Bures, UnbalancedBures

def _validate_losses(loss: Union[str, Sequence[ArrayLike]],
                     subsets: Sequence[Any]):
    if isinstance(loss, str):
        # TODO(@MUCDK) check for valid strings of losses. this does depend on backend
        return [loss] * len(subsets)
    elif isinstance(loss, Sequence):
        if len(subsets) != len(loss):
            raise ValueError("The number of cost matrices provided must be equal to the number of OT problems.")


def _validate_loss(loss: Union[str, ArrayLike],
                   adata_source: Optional[AnnData],
                   adata_target: Optional[AnnData]):
    if isinstance(loss, str):
        # TODO(@MUCDK) check for valid strings of losses. this does depend on backend
        return None
    elif isinstance(loss, ArrayLike):
        if adata_source.n_obs != len(loss.shape[0]):
            raise ValueError(f"Dimension mismatch. The cost matrix provided has dimension {loss.shape} "
                             f"but the number of data points for the source distribution is {adata_source.n_obs}.")
        if adata_target.n_obs != len(loss.shape[1]):
            raise ValueError(f"Dimension mismatch. The cost matrix provided has dimension {loss.shape} "
                             f"but the number of data points for the source distribution is {adata_target.n_obs}.")


def _get_marginal(adata: AnnData,
                  attr: Optional[str] = None,
                  key: Optional[str] = None) -> ArrayLike:
    if attr is None:
        return np.ones(adata.n_obs) / adata.n_obs

    if not hasattr(adata, attr):
        raise AttributeError("TODO: invalid attribute")
    container = getattr(adata, attr)

    if key is None:
        return np.array(_normalize(container))
    if key not in container:
        raise KeyError(f"TODO: unable to find `adata.{attr}['{key}']`.")
    return _normalize(np.array(container[key]))


def _normalize(arr: ArrayLike) -> ArrayLike:
    if arr.ndim != 1:
        raise ValueError("TODO: expected 1D")
    return arr/np.sum(arr)


def _get_backend_losses(backend: str = "JAX", **kwargs: Any):
    if backend == "JAX":
        dimension = kwargs.pop("dimension", 1)
        return {"Euclidean": Euclidean(**kwargs),
                "Cosine": Cosine(**kwargs),
                "Bures": Bures(dimension, **kwargs),
                "UnbalancedBures": UnbalancedBures(dimension, **kwargs)}
    else:
        raise NotImplementedError()