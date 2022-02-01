from typing import Any, Union, Optional, Sequence, Mapping


from numpy.typing import ArrayLike
from ott.geometry.costs import Bures, Cosine, Euclidean, UnbalancedBures
import numpy as np
import numpy.typing as npt
from anndata import AnnData


def _validate_losses(loss: Union[str, Sequence[ArrayLike]], subsets: Sequence[Any]):
    if isinstance(loss, str):
        # TODO(@MUCDK) check for valid strings of losses. this does depend on backend
        return [loss] * len(subsets)
    elif isinstance(loss, Sequence):
        if len(subsets) != len(loss):
            raise ValueError("The number of cost matrices provided must be equal to the number of OT problems.")


def _validate_loss(loss: Union[str, ArrayLike], adata_source: Optional[AnnData], adata_target: Optional[AnnData]):
    if isinstance(loss, str):
        # TODO(@MUCDK) check for valid strings of losses. this does depend on backend
        return None
    elif isinstance(loss, ArrayLike):
        if adata_source.n_obs != len(loss.shape[0]):
            raise ValueError(
                f"Dimension mismatch. The cost matrix provided has dimension {loss.shape} "
                f"but the number of data points for the source distribution is {adata_source.n_obs}."
            )
        if adata_target.n_obs != len(loss.shape[1]):
            raise ValueError(
                f"Dimension mismatch. The cost matrix provided has dimension {loss.shape} "
                f"but the number of data points for the source distribution is {adata_target.n_obs}."
            )


def _get_marginal(adata: AnnData, attr: Optional[str] = None, key: Optional[str] = None) -> ArrayLike:

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


def _verify_dict(adata: AnnData, d: dict):
    if "attr" not in d.keys():
        raise ValueError(
            "Please provide the item with key 'attr'.")
    if not hasattr(adata, d["attr"]):
        raise AttributeError("TODO: invalid attribute")
    if "key" not in d.keys():
        raise AttributeError("TODO: provide 'attr' and 'key' as keys for this dict")
    else:
        if not hasattr(getattr(adata, d["attr"]), d["key"]):
            raise AttributeError("TODO: invalid key of attribute")

def _verify_marginals(adata: AnnData, marginals: Optional[Union[Sequence[Union[Mapping[str, Any], npt.ArrayLike]], Mapping[str, Any], npt.ArrayLike]]):

    if isinstance(marginals, Sequence):
        for marg in marginals:
            if isinstance(marg, Mapping):
                _verify_dict(adata, marg)
            elif isinstance(marg, npt.ArrayLike):
                pass
            else:
                raise ValueError(
                    "The marginals must be given as npt.ArrayLike or as a Mapping pointing to their locations.")
    else:
        if isinstance(marginals, Mapping):
            _verify_dict(adata, marginals)
        elif isinstance(marginals, npt.ArrayLike):
            pass
        else:
            raise ValueError(
                "The marginals must be given as npt.ArrayLike or as a Mapping pointing to their locations.")

def _get_differences(items):
    diff_dict = {}
    for tup in items:
        try:
            delta_t = np.float(tup[1]) - np.float(tup[0])
            diff_dict[tup] = delta_t
        except ValueError:
            print("The values {} of the time column cannot be interpreted as floats.".format(tup))
    return diff_dict

    
def _normalize(arr: ArrayLike) -> ArrayLike:
    if arr.ndim != 1:
        raise ValueError("TODO: expected 1D")
    return arr / np.sum(arr)


# TODO: use the enum in backend
def _get_backend_losses(backend: str = "JAX", **kwargs: Any):
    if backend == "JAX":
        dimension = kwargs.pop("dimension", 1)
        return {
            "Euclidean": Euclidean(**kwargs),
            "Cosine": Cosine(**kwargs),
            "Bures": Bures(dimension, **kwargs),
            "UnbalancedBures": UnbalancedBures(dimension, **kwargs),
        }
    else:
        raise NotImplementedError()
